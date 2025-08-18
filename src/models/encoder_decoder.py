import copy
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from types import SimpleNamespace

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.nn import CrossEntropyLoss
from torchmetrics import MaxMetric, MeanMetric, SpearmanCorrCoef
from transformers import T5EncoderModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.tokenization_utils_base import BatchEncoding

##### define reconction model
class ReconstructionModel(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, max_seq_length: int):
        super().__init__()
        # Project embedding to sequence of hidden states
        self.proj = nn.Linear(hidden_size, hidden_size * max_seq_length)
        self.norm = nn.LayerNorm(hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length

    def forward(self, z):
        # z: [batch_size, hidden_size]
        batch_size = z.size(0)
        # Project to [batch_size, max_seq_length * hidden_size]
        seq = self.proj(z)
        # Reshape to [batch_size, max_seq_length, hidden_size]
        seq = seq.view(batch_size, self.max_seq_length, -1)
        # Normalize
        seq = self.norm(seq)
        # Project to logits: [batch_size, max_seq_length, vocab_size]
        logits = self.out(seq)
        return logits

    def model(self):
        return self

#### define encoder-decoder module
class EncoderDecoderModule(LightningModule):
    def __init__(
        self,
        encoder_model: T5EncoderModel,
        decoder_model: Callable[..., T5Stack],
        rec_model : Any = None,
        input_tokenizer: Any,
        output_tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        compile: bool,
        scheduler=None,
        metadata_embedder: Optional[nn.Module] = None,
        metadata_embedder_output_dim: Optional[int] = None,
        non_shuffled_datamodule=None,
        temperature: float = 0.07,
        max_seq_length: int = 512,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        ##### define encoder
        self.encoder_hidden_size = encoder_model.config.hidden_size
        self.decoder_hidden_size = self.encoder_hidden_size
        self.non_shuffled_datamodule = non_shuffled_datamodule
        self.metadata_embedder = metadata_embedder

        self.encoder = encoder_model.encoder
        self.shared = encoder_model.shared
        ##### define decoder
        decoder_config = decoder_model.keywords["config"]
        decoder_config.vocab_size = output_tokenizer.vocab_size
        decoder_config.d_model = encoder_model.config.d_model
        decoder_config.is_decoder = True
        decoder_config.use_cache = True
        decoder_config.add_cross_attention = True

        self.decoder = decoder_model()
        self.temperature = temperature
        #### define reconstruction layer
        self.rec_model = rec_model.model() if rec_model else ReconstructionModel(hidden_size=self.encoder_hidden_size,
                vocab_size=input_tokenizer.vocab_size,
                max_seq_length=max_seq_length)


        ###################################
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size),
            nn.ReLU(),
            nn.Linear(self.encoder_hidden_size, 128),
        )
        #### preject meta data
        self.metadata_projection_head = nn.Sequential(
            nn.Linear(metadata_embedder_output_dim, metadata_embedder_output_dim),
            nn.ReLU(),
            nn.Linear(metadata_embedder_output_dim, 128),
        )
        #### define decoder input projection 
        self.decoder_input_proj = nn.Linear(
            self.encoder_hidden_size, self.decoder_hidden_size
        )

        self.lm_head = nn.Linear(
            self.decoder_hidden_size, decoder_config.vocab_size, bias=False
        )

        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer

        self.train_total_loss = MeanMetric()
        self.train_loss = MeanMetric()
        self.train_con_loss = MeanMetric()
        self.train_lip_loss = MeanMetric()
        self.train_rec_loss = MeanMetric()

        self.val_total_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_con_loss = MeanMetric()
        self.val_lip_loss = MeanMetric()
        self.val_rec_loss = MeanMetric()

    ### define reconstruction loss 
    def rec_loss(self, x: torch.Tensor, recon_outputs: torch.Tensor) -> torch.Tensor:
        loss = torch.nn.functional.cross_entropy(
               recon_outputs.view(-1, recon_outputs.size(-1)),  # [batch_size * max_seq_length, vocab_size]
               x.to(self.device).view(-1),                     # [batch_size * max_seq_length]
               ignore_index=0,                                 # Ignore padding token
               reduction="mean"
               ) / self.temperature
        return loss

    def contrastive_loss(self, embeddings1, embeddings2):
        embeddings1 = F.normalize(embeddings1, dim=1)
        embeddings2 = F.normalize(embeddings2, dim=1)

        metadata_sim = torch.matmul(embeddings2, embeddings2.T)
        metadata_sim_max, _ = metadata_sim.max(dim=1, keepdim=True)
        metadata_sim_min, _ = metadata_sim.min(dim=1, keepdim=True)
        metadata_sim = (metadata_sim - metadata_sim_min) / (
            metadata_sim_max - metadata_sim_min + 1e-10
        )

        similarity_matrix = torch.matmul(embeddings1, embeddings1.T)
        similarity_matrix = similarity_matrix / self.temperature

        log_prob = F.log_softmax(similarity_matrix, dim=1)

        mask = ~torch.eye(
            embeddings1.shape[0], dtype=torch.bool, device=embeddings1.device
        )
        loss = -(metadata_sim[mask] * log_prob[mask]).mean()

        return loss

    def lipschitz_loss(self, z, y):
        if torch.all(y == y[0]):
            dif_z = torch.sqrt(
                torch.sum((z.unsqueeze(1) - z.unsqueeze(0)) ** 2, dim=2) + 1e-10
            )
            return dif_z.mean()
        
        dif_y = (y.unsqueeze(1) - y.unsqueeze(0)).squeeze(-1)
        dif_z = torch.sqrt(
            torch.sum((z.unsqueeze(1) - z.unsqueeze(0)) ** 2, dim=2) + 1e-10
        )
        
        lips = abs(dif_y / (dif_z + 1e-10))
        ratio = lips - torch.median(lips)
        ratio = ratio[ratio > 0]
        
        if len(ratio) == 0:
            return torch.tensor(0.0, device=self.device)
            
        return ratio.mean()
    #### define forward
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Encoder
        input_embeds = self.shared(input_ids)
        encoder_outputs = self.encoder(
            inputs_embeds=input_embeds, attention_mask=attention_mask
        )

        encoder_hidden_states = encoder_outputs.last_hidden_state

        mean_pooled = self._mean_pooling(encoder_hidden_states, attention_mask)
        projected_embeddings = self.projection_head(mean_pooled)
        # Reconstruction 
        recon_outputs = self.rec_model(mean_pooled)
        rec_loss = self.rec_loss(input_ids, recon_outputs) if input_ids is not None else None



        # Decoder
        if decoder_input_ids is not None:
            decoder_inputs = self.shared(decoder_input_ids)
            decoder_inputs = self.decoder_input_proj(decoder_inputs)

        decoder_outputs = self.decoder(
            inputs_embeds=decoder_inputs,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        ).last_hidden_state

        lm_logits = self.lm_head(decoder_outputs)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            
        return SimpleNamespace(
            loss=loss,
            logits=lm_logits,
            encoder_hidden_states=encoder_hidden_states,
            projected_embeddings=projected_embeddings,
            rec_loss= rec_loss
        )
    ### hook function begin each batch
    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_total_loss.reset()
        self.val_con_loss.reset()
        self.val_lip_loss.reset()
        self.val_rec_loss.reset()
    #### return loss 
    def model_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        return loss, outputs, batch["labels"]
    #### training 
    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        try:
            non_shuffled_batch = next(self.non_shuffled_train_iter)
        except:
            self.non_shuffled_train_iter = iter(
                self.non_shuffled_datamodule.train_dataloader()
            )
            non_shuffled_batch = next(self.non_shuffled_train_iter)

        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
        )
        #### prediction loss
        main_loss = outputs.loss
        #### rec loss
        rec_loss = outputs.rec_loss
        ##### compute contrastive loss with metadata 
        with torch.no_grad():
            m_embeddings = self._emb_metadata(batch["metadata"])
        metadata_embeddings = self.metadata_projection_head(m_embeddings)
        contrastive_loss = self.contrastive_loss(
            outputs.projected_embeddings, 
            metadata_embeddings
        )

        non_shuffled_outputs = self.encoder(
            input_ids=non_shuffled_batch["input_ids"].to(self.device),
            attention_mask=non_shuffled_batch["attention_mask"].to(self.device),
        ).last_hidden_state
        non_shuffled_emb = self._mean_pooling(non_shuffled_outputs, non_shuffled_batch["attention_mask"].to(self.device))
        #### create task data: embedding-value
        task_data = {}
        for i in range(len(non_shuffled_batch["task_name"])):
            task = non_shuffled_batch["task_name"][i]
            if task not in task_data:
                task_data[task] = {"text": [], "value": []}

            task_data[task]["text"].append(non_shuffled_emb[i])
            if non_shuffled_batch["value"].dim() > 1:
                task_data[task]["value"].append(non_shuffled_batch["value"][i : i + 1])
            else:
                task_data[task]["value"].append(non_shuffled_batch["value"][i])
        #### compute lipschitz_loss 
        lipschitz_loss = 0
        for task in task_data:
            if len(task_data[task]["text"]) > 0:
                task_data[task]["text"] = torch.stack(task_data[task]["text"], dim=0)

            if len(task_data[task]["value"]) > 0:
                try:
                    task_data[task]["value"] = torch.cat(
                        task_data[task]["value"], dim=0
                    )
                except:
                    task_data[task]["value"] = torch.tensor(task_data[task]["value"])

            lipschitz_loss += self.lipschitz_loss(
                z=task_data[task]["text"].to(self.device), y=task_data[task]["value"].to(self.device)
            ) * (len(task_data[task]["value"]) / len(non_shuffled_batch["value"]))
        
        #### Add rec loss to total loss
        total_loss = main_loss + rec_loss + contrastive_loss / (contrastive_loss / main_loss).detach() + lipschitz_loss / (lipschitz_loss / main_loss).detach()
        # total_loss = main_loss + contrastive_loss + lipschitz_loss
       
        self.train_total_loss(total_loss)
        self.train_loss(main_loss)
        self.train_con_loss(contrastive_loss)
        self.train_lip_loss(lipschitz_loss)
        self.train_rec_loss(rec_loss)

        self.log("train/main_loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/total_loss", self.train_total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/contrastive_loss", self.train_con_loss, on_step=True, on_epoch=True)
        self.log("train/lipschitz_loss", self.train_lip_loss, on_step=True, on_epoch=True)
        self.log("train/rec_loss", self.train_rec_loss, on_step = True, on_epoch = True)


        return total_loss

    def on_train_epoch_end(self) -> None:
        pass

    def _emb_metadata(self, m: Tuple[BatchEncoding]) -> torch.Tensor:
        with torch.no_grad():
            encoded_input = m.to(self.device)
            emb_m = self.metadata_embedder(**encoded_input)
            emb_m = emb_m.last_hidden_state
            emb_m = self._mean_pooling(emb_m, encoded_input["attention_mask"])
        return emb_m
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        try:
            non_shuffled_batch = next(self.non_shuffled_train_iter)
        except:
            self.non_shuffled_train_iter = iter(
                self.non_shuffled_datamodule.train_dataloader()
            )
            non_shuffled_batch = next(self.non_shuffled_train_iter)

        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
        )
        ### get main loss and rec loss
        main_loss = outputs.loss
        rec_loss = outputs.rec_loss

        with torch.no_grad():
            m_embeddings = self._emb_metadata(batch["metadata"])
        metadata_embeddings = self.metadata_projection_head(m_embeddings)
        contrastive_loss = self.contrastive_loss(
            outputs.projected_embeddings, 
            metadata_embeddings
        )

        non_shuffled_outputs = self.encoder(
            input_ids=non_shuffled_batch["input_ids"].to(self.device),
            attention_mask=non_shuffled_batch["attention_mask"].to(self.device),
        ).last_hidden_state
        non_shuffled_emb = self._mean_pooling(non_shuffled_outputs, non_shuffled_batch["attention_mask"].to(self.device))
        
        task_data = {}
        for i in range(len(non_shuffled_batch["task_name"])):
            task = non_shuffled_batch["task_name"][i]
            if task not in task_data:
                task_data[task] = {"text": [], "value": []}

            task_data[task]["text"].append(non_shuffled_emb[i])
            if non_shuffled_batch["value"].dim() > 1:
                task_data[task]["value"].append(non_shuffled_batch["value"][i : i + 1])
            else:
                task_data[task]["value"].append(non_shuffled_batch["value"][i])

        lipschitz_loss = 0
        for task in task_data:
            if len(task_data[task]["text"]) > 0:
                task_data[task]["text"] = torch.stack(task_data[task]["text"], dim=0)

            if len(task_data[task]["value"]) > 0:
                try:
                    task_data[task]["value"] = torch.cat(
                        task_data[task]["value"], dim=0
                    )
                except:
                    task_data[task]["value"] = torch.tensor(task_data[task]["value"])

            lipschitz_loss += self.lipschitz_loss(
                z=task_data[task]["text"].to(self.device), y=task_data[task]["value"].to(self.device)
            ) * (len(task_data[task]["value"]) / len(non_shuffled_batch["value"]))

        # total_loss = main_loss + contrastive_loss / (contrastive_loss / main_loss).detach() + lipschitz_loss / (lipschitz_loss / main_loss).detach()
        # Add rec_loss to total loss
        total_loss = main_loss + rec_loss + contrastive_loss + lipschitz_loss

        self.val_total_loss(total_loss)
        self.val_loss(main_loss)
        self.val_con_loss(contrastive_loss)
        self.val_lip_loss(lipschitz_loss)
        self.val_rec_loss(rec_loss)

        self.log("val/main_loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/total_loss", self.val_total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/contrastive_loss", self.val_con_loss, on_step=True, on_epoch=True)
        self.log("val/lipschitz_loss", self.val_lip_loss, on_step=True, on_epoch=True)
        self.log("val/rec_loss", self.val_rec_loss, on_step = True, on_epoch = True)
        
        return total_loss

    def on_validation_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.non_shuffled_datamodule.setup("fit")
            loader = self.non_shuffled_datamodule.train_dataloader()
            self.non_shuffled_train_iter = iter(loader)

            self.metadata_embedder = torch.compile(self.metadata_embedder)
            self.metadata_projection_head = torch.compile(self.metadata_projection_head)
            self.projection_head = torch.compile(self.projection_head)

            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)
            self.shared = torch.compile(self.shared)
            self.decoder_input_proj = torch.compile(self.decoder_input_proj)
            self.lm_head = torch.compile(self.lm_head)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())

        if self.hparams.scheduler is not None:
            # Calculate total steps
            total_steps = self.trainer.estimated_stepping_batches

            scheduler = self.hparams.scheduler(
                optimizer=optimizer, num_training_steps=total_steps
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # Update lr every step instead of epoch
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 32,
        temperature: float = 0.7,
        top_k: int = 20,
        top_p: float = 0.95,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]

        # Encoder
        input_embeds = self.shared(input_ids)
        encoder_outputs = self.encoder(
            inputs_embeds=input_embeds, attention_mask=attention_mask
        ).last_hidden_state

        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.output_tokenizer.bos_token_id,
            dtype=torch.long,
            device=self.device,
        )

        past_key_values = None

        for i in range(max_length - 1):
            if i == 0:
                decoder_inputs = self.shared(decoder_input_ids)
                decoder_inputs = self.decoder_input_proj(decoder_inputs)
            else:
                last_token = decoder_input_ids[:, -1:]
                decoder_inputs = self.shared(last_token)
                decoder_inputs = self.decoder_input_proj(decoder_inputs)

            # cache for acceleration
            decoder_outputs = self.decoder(
                inputs_embeds=decoder_inputs,
                encoder_hidden_states=encoder_outputs,
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = decoder_outputs.past_key_values
            hidden_states = decoder_outputs.last_hidden_state

            next_token_logits = self.lm_head(hidden_states[:, -1:]) / temperature
            next_token_logits = next_token_logits.squeeze(1)

            if i == 0:
                scores = torch.empty_like(next_token_logits)

            # Top-K + Top-P
            if top_k > 0 or top_p < 1.0:
                scores.copy_(next_token_logits)

                if top_k > 0:
                    indices_to_remove = (
                        scores < torch.topk(scores, top_k)[0][..., -1, None]
                    )
                    scores[indices_to_remove] = float("-inf")

                if top_p < 1.0:
                    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_scores, dim=-1), dim=-1
                    )

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = torch.zeros_like(
                        scores, dtype=torch.bool
                    ).scatter_(1, sorted_indices, sorted_indices_to_remove)
                    scores[indices_to_remove] = float("-inf")

            probs = F.softmax(scores, dim=-1)
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)

            if (next_token == self.output_tokenizer.eos_token_id).all():
                break

        return decoder_input_ids

    @torch.inference_mode()
    def generate_numbers(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 32,
        decoding_strategy: str = "top_k",  # Add decoding strategy parameter
        **kwargs,
    ):
        if decoding_strategy != "top_k":
            raise NotImplementedError

        predictions = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            decoding_strategy=decoding_strategy,
            **kwargs,
        )

        pred_numbers = []
        for pred in predictions:
            # print("Token IDs:", pred[pred != -100].tolist())

            # tokens = [
            #     self.output_tokenizer._convert_id_to_token(idx)
            #     for idx in pred[pred != -100]
            # ]
            # print("Tokens:", tokens)

            text = self.output_tokenizer.decode(
                pred[pred != -100], skip_special_tokens=True
            )
            # print("Decoded text:", text)

            try:
                num = float(text)
                pred_numbers.append(num)
            except ValueError:
                if len(pred_numbers) > 0:
                    mean_value = sum(pred_numbers) / len(pred_numbers)
                    pred_numbers.append(mean_value)
                else:
                    pred_numbers.append(0.0)
            # print("---")

        # Return a two-dimensional vector
        return torch.tensor(pred_numbers, device=self.device).reshape(-1, 1)
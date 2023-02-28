import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, Type, Callable, List, Any

import torch
import torch.nn as nn
import captum
from torch.nn import CrossEntropyLoss

from transformers import Trainer


class AttributionTrainer(Trainer):

    def __init__(self, attribution_method, **kwargs):
        super(AttributionTrainer, self).__init__(**kwargs)
        self.attribution_method = attribution_method

    def compute_loss(self, model, inputs, **kwargs):
        outputs = {
            'attributions': self.attribution_method.attribute(inputs),
        }
        loss = torch.tensor(-1, dtype=torch.float, device=inputs['labels'].device)
        return loss, outputs


@dataclass
class AttributionArguments:
    pass


class Attributer:

    def __init__(self, attribution_args, training_args, model, tokenizer, data_collator):
        last_period = training_args.captum_class.rfind('.')
        class_name = training_args.captum_class[last_period + 1:]
        module_name = training_args.captum_class[:last_period]
        module = importlib.import_module(module_name)
        attribution_method = getattr(module, class_name)(model, tokenizer)

        self.trainer = AttributionTrainer(
            attribution_method,
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    def attribute(self, dataset):
        return self.trainer.predict(dataset)


class Processing:

    @staticmethod
    def mean_embed_dims(attributions):
        return attributions.mean(dim=-1)

    @staticmethod
    def normalize_0_p1(attributions):
        return 0.5 + Processing.normalize_n1_p1(attributions) / 2

    @staticmethod
    def normalize_n1_p1(attributions):
        return (attributions - attributions.mean()) / attributions.std()

    @staticmethod
    def rescale_0_p1(attributions):
        return attributions / attributions.abs().max()

    @staticmethod
    def rescale_sum1(attributions):
        return attributions / (attributions.sum() + 1e-8)

    @staticmethod
    def softmax(attributions):
        return nn.functional.softmax(attributions)


def create_token_baseline(raw_inputs, token_embedding):
    input_length = raw_inputs['attention_mask'].size()[1]
    token_embedding = token_embedding.unsqueeze(0).expand(-1, input_length, -1)
    return token_embedding


class AttributionMethodWrap(ABC, nn.Module):
    _captum_cls: Type
    _processing_steps: List[Callable] = []
    _loss_func_kwargs = {}

    _method_args: Any
    _method_kwargs: Any

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.captum_object = self._captum_cls(self)

    @abstractmethod
    def forward(self, inputs, kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _prepare_inputs(self, raw_inputs):
        raise NotImplementedError()

    def _calc_loss(self, labels, outputs):
        logits = outputs['logits']
        if self.model.config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss(**self._loss_func_kwargs)
            loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))
        else:
            raise NotImplementedError()
        return loss.view(1) if len(loss.size()) == 0 else loss

    def _method_specific_args(self, raw_inputs):
        return [], {}

    def _process_results(self, results):
        for process_step in self._processing_steps:
            results = process_step(results)
        return results

    def attribute(self, raw_inputs):
        prepared_inputs, other_inputs = self._prepare_inputs(raw_inputs)
        args, kwargs = self._method_specific_args(raw_inputs)
        self._method_args = args
        self._method_kwargs = kwargs
        results = self.captum_object.attribute(prepared_inputs, *args, additional_forward_args=(other_inputs,), **kwargs)
        results = self._process_results(results)
        return results


class FeatureLevelMethodWrap(AttributionMethodWrap, ABC):
    _processing_steps = [Processing.mean_embed_dims]

    def forward(self, inputs, kwargs):
        internal_batch_size = inputs.shape[0]  # how often captum repeated the input
        kwargs['labels'] = kwargs['labels'].expand((internal_batch_size,))
        kwargs['attention_mask'] = kwargs.attention_mask.expand((internal_batch_size, -1))
        outputs = self.model(inputs_embeds=inputs, **kwargs)
        return self._calc_loss(kwargs['labels'], outputs)

    def _prepare_inputs(self, raw_inputs):
        input_ids = raw_inputs.pop('input_ids')
        embeddings = self.model.get_input_embeddings()
        inputs_embeds = embeddings(input_ids)
        return inputs_embeds, raw_inputs


class TokenLevelMethodWrap(AttributionMethodWrap, ABC):
    _custom_baseline_embedding: torch.FloatTensor = None

    def forward(self, inputs: torch.Tensor, kwargs):
        internal_batch_size = inputs.shape[0]  # how often captum repeated the input
        kwargs['labels'] = kwargs['labels'].expand((internal_batch_size,))
        kwargs['attention_mask'] = kwargs.attention_mask.expand((internal_batch_size, -1))
        if self._custom_baseline_embedding is not None:
            embeddings = self.model.get_input_embeddings()
            input_embeds = embeddings(inputs)
            mask = (inputs == self._method_kwargs['baselines']).unsqueeze(-1).expand_as(input_embeds)
            input_embeds[mask] = self._custom_baseline_embedding.to(input_embeds).expand_as(input_embeds[mask])
            outputs = self.model(inputs_embeds=input_embeds, **kwargs)
        else:
            outputs = self.model(input_ids=inputs, **kwargs)
        return self._calc_loss(kwargs['labels'], outputs)

    def _prepare_inputs(self, raw_inputs):
        input_ids = raw_inputs.pop('input_ids')
        return input_ids, raw_inputs


class IntegratedGradients(FeatureLevelMethodWrap):
    _captum_cls = captum.attr.IntegratedGradients
    _processing_steps = [
        Processing.mean_embed_dims,
        Processing.rescale_sum1,
    ]

    def _method_specific_args(self, raw_inputs):
        return [], {'internal_batch_size': 10}           # TODO make this a configuration parameter
    #
    # def _method_specific_args(self, raw_inputs):
    #     device = raw_inputs['attention_mask'].device
    #     embeddings = self.model.get_input_embeddings()
    #     token_id = torch.tensor([self.tokenizer.mask_token_id], dtype=torch.long, device=device)
    #     token_embedding = embeddings(token_id) / 2
    #     #token_embedding = embeddings.weight.mean(0, keepdim=True)
    #     #token_embedding = torch.full_like(token_embedding, 0.01)
    #     return [], {
    #         'baselines': create_token_baseline(raw_inputs, token_embedding),
    #         'internal_batch_size': 25,
    #     }


class SaliencyAbs(FeatureLevelMethodWrap):
    _captum_cls = captum.attr.Saliency
    _processing_steps = [
        Processing.mean_embed_dims,
        Processing.rescale_sum1,
    ]


class Saliency(FeatureLevelMethodWrap):
    _captum_cls = captum.attr.Saliency
    _processing_steps = [
        Processing.mean_embed_dims,
        Processing.rescale_sum1,
    ]

    def _method_specific_args(self, raw_inputs):
        return [], {'abs': False}


class DeepLift(FeatureLevelMethodWrap):
    _captum_cls = captum.attr.DeepLift
    _processing_steps = [
        Processing.mean_embed_dims,
        Processing.rescale_sum1,
    ]
    _loss_func_kwargs = {'reduction': 'none'}

    # def _method_specific_args(self, raw_inputs):
    #     device = raw_inputs['attention_mask'].device
    #     embeddings = self.model.get_input_embeddings()
    #     mask_token_id = torch.tensor([self.tokenizer.mask_token_id], dtype=torch.long, device=device)
    #     return [], {
    #         'baselines': create_token_baseline(raw_inputs, embeddings(mask_token_id))
    #     }


class DeepLiftShap(FeatureLevelMethodWrap):
    _captum_cls = captum.attr.DeepLiftShap

    def _method_specific_args(self, raw_inputs):
        # TODO implement baseline distribution
        raise NotImplementedError()


class GradientShap(FeatureLevelMethodWrap):
    _captum_cls = captum.attr.GradientShap

    def _method_specific_args(self, raw_inputs):
        # TODO implement baseline distribution
        raise NotImplementedError()


class InputXGradient(FeatureLevelMethodWrap):
    _captum_cls = captum.attr.InputXGradient
    _processing_steps = [
        Processing.mean_embed_dims,
        Processing.rescale_sum1,
    ]


class GuidedBackprop(FeatureLevelMethodWrap):
    _captum_cls = captum.attr.GuidedBackprop
    _processing_steps = [
        Processing.mean_embed_dims,
        Processing.rescale_sum1,
    ]
    _loss_func_kwargs = {'reduction': 'none'}


class Lime(TokenLevelMethodWrap):
    _captum_cls = captum.attr.Lime
    _processing_steps = [
        Processing.rescale_sum1
    ]

    def _method_specific_args(self, raw_inputs):
        token_id = self.tokenizer.mask_token_id
        return [], {'baselines': token_id}


class KernelSHAP(TokenLevelMethodWrap):
    _captum_cls = captum.attr.KernelShap
    _processing_steps = [
        Processing.rescale_sum1
    ]

    def _method_specific_args(self, raw_inputs):
        token_id = self.tokenizer.mask_token_id
        return [], {
            'n_samples': 50,
            'baselines': token_id
        }

import logging

from toxic_x_dom.data import load_toxic_span_datasets

from datasets import Dataset, ClassLabel


logger = logging.getLogger(__name__)


def load_dataset(data_args, tokenizer):

    class_label = ClassLabel(names=['I', 'O', 'B'])
    I, O, B = class_label.str2int(['I', 'O', 'B'])

    def add_iob_labels(sample, batch_encoding):
        mask = sample['toxic_mask']

        labels = []
        for token_idx, input_id in enumerate(batch_encoding['input_ids']):
            char_span = batch_encoding.token_to_chars(0, token_index=token_idx)
            if char_span is None:  # special token
                labels.append(-100)
                continue
            token_toxicity = sum(mask[char_span.start:char_span.end]) / float(char_span.end - char_span.start)
            token_toxic = token_toxicity > 0.5
            if token_toxic:
                if token_idx == 0 or labels[token_idx - 1] not in {B, I}:
                    labels.append(B)
                else:
                    labels.append(I)
            else:
                labels.append(O)
        batch_encoding['labels'] = labels
        return batch_encoding

    pandas_datasets = load_toxic_span_datasets()
    dataset = pandas_datasets[data_args.dataset_name]

    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    hf_dataset = Dataset.from_pandas(dataset)
    hf_dataset = hf_dataset.map(
        lambda sample: add_iob_labels(
            sample,
            tokenizer(
                sample['full_text'],
                padding=padding,
                truncation=True,
                max_length=data_args.max_seq_length,
            )
        )
    )

    #TODO remove unnecessary columns

    return hf_dataset

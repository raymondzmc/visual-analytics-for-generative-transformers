import os
import pdb
import json
import torch
import argparse
from umap import umap_ as UMAP

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
from scipy import linalg as la
from utils.head_importance import get_head_importance_pegasus
from utils.helpers import normalize, format_attention, format_attention_image
from os.path import join as pjoin
import chinese_converter


def main(args):
    device = args.device
    model_name = args.model_name
    # dataset = load_dataset('xsum')['test']

    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh").to(device)
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    with open("../resources/opus-2020-07-17.test.txt", "r") as f:
        lines = f.readlines()

    dataset = []
    for line in lines:
        if line.startswith(">>yue_Hant<<"):
            dataset.append(line)

    # pdb.set_trace()
    # model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    hidden_size = model.model.config.hidden_size
    max_output_len = 512
    max_input_len = model.config.max_position_embeddings

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    encoder_hiddens = np.zeros((len(dataset), hidden_size), dtype=np.float16)
    decoder_hiddens = []
    decoder_len = []
    encoder_len = []
    model.eval()

    num_steps = 0

    for i, example in enumerate(tqdm(dataset)):
        _id = i
        batch = tokenizer([example], truncation=True, padding="longest", return_tensors="pt").to(
            device
        )
        batch["max_length"] = max_output_len
        batch["return_dict_in_generate"] = True
        batch["output_attentions"] = False
        batch["output_hidden_states"] = True
        batch["output_scores"] = True

        with torch.no_grad():
            output = model.generate(**batch)

        prediction = tokenizer.decode(output["sequences"][0])
        chinese_converter.to_simplified(prediction)

        beam_indices = output["beam_indices"][0, :-1]

        input_len = len(batch.input_ids[0])
        encoder_len.append(input_len)
        output_len = len(beam_indices)

        encoder_hidden_states = output.encoder_hidden_states[-1][0].mean(0)
        encoder_hiddens[i] = encoder_hidden_states.half().cpu().numpy()

        # output_len x (1 + n_layer) x beam_size x batch_size x hidden_dim
        decoder_hidden_states = output.decoder_hidden_states
        decoder_hidden_states = torch.stack([hidden[-1][:, 0] for hidden in decoder_hidden_states])[
            :output_len
        ]
        beam_search_hidden_states = []
        for step, hidden in enumerate(decoder_hidden_states):
            beam_idx = beam_indices[step].item()
            beam_search_hidden_states.append(hidden[beam_idx])

        decoder_hiddens.append(torch.stack(beam_search_hidden_states).half().cpu().numpy())

        num_steps += 1

    torch.save(encoder_hiddens, pjoin(args.output_dir, "encoder_hidden_states.pt"))
    torch.save(decoder_hiddens, pjoin(args.output_dir, "decoder_hidden_states.pt"))

    fit = UMAP.UMAP(
        n_neighbors=5,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
        verbose=True,
        low_memory=True,
        init="random",
    )

    encoder_projections = fit.fit_transform(encoder_hiddens)
    torch.save(encoder_projections, pjoin(args.output_dir, "encoder_mean_projections.pt"))
    decoder_len = [len(h) for h in decoder_hiddens]

    decoder_all_hiddens = np.concatenate(decoder_hiddens, axis=0)
    decoder_all_projections = fit.fit_transform(decoder_all_hiddens)
    _decoder_all_projections = []
    start = 0
    for output_len in decoder_len:
        _decoder_all_projections.append(
            decoder_all_projections[start : start + output_len].tolist()
        )
        start = start + output_len
    torch.save(_decoder_all_projections, pjoin(args.output_dir, "decoder_projections.pt"))

    # pdb.set_trace()
    encoder_projection_data = {
        "x": encoder_projections[:, 0].tolist(),
        "y": encoder_projections[:, 1].tolist(),
        "ids": np.arange(len(encoder_projections)),
        "continuous": {
            "input_len": np.array(encoder_len),
        },
    }
    torch.save(encoder_projection_data, os.path.join(args.output_dir, "encoder_projection_data.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarization")
    parser.add_argument("-dataset", type=str, default="xsum", choices=["cnndm", "xsum"])
    parser.add_argument("-model_name", type=str, default="google/pegasus-xsum")
    parser.add_argument("-hidden_aggregate_method", type=str, default="mean")
    parser.add_argument("-output_dir", type=str, default="resources/nmt")

    args = parser.parse_args()
    # args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = torch.device("cuda:0")
    main(args)

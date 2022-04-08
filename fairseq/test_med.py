from ast import arg
import json
import torch
import os
import argparse
from fairseq.models.roberta import RobertaModel
from examples.roberta import med_qa  # load the Commonsense QA task

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="Path to pre-trained model")
    parser.add_argument(
            "--knowledge_layer",
            nargs='+',
            default=[-1, 0],
            help="Layers that would add kowledge embedding",
        )
    parser.add_argument(
            "--data_file",
            type=str,
            default='../data/openbook_k/test.jsonl',
            help="The data to be evaluate",
        )
    args = parser.parse_args()
    roberta = RobertaModel.from_pretrained(args.model_path, 'checkpoint_best.pt', '../data/med_qa/', knowledge_layer=args.knowledge_layer)
    print(roberta)
    roberta.eval()  # disable dropout
    roberta.cuda()  # use the GPU (optional)
    nsamples, ncorrect = 0, 0
    max_know_length = 256
    corrects = []
    with torch.no_grad():
        with open(args.data_file) as h:
            for line in h:
                example = json.loads(line)
                scores = []
                for choice in example['choices']:
                    knowledges = choice["knowledge"]
                    knowledge_text = ""
                    for know in knowledges:
                        knowledge_text += know 
                    input = roberta.encode(
                        'Q: ' + example['question'],
                        'A: ' + choice['text'],
                        # knowledge_text
                        no_separator=True,
                        truncate=True,
                    )
                    
                    know_bin = []
                    for know in knowledges:
                        bina_know = roberta.encode(know, not_a_sentence=True)
                        bina_know = bina_know[:max_know_length]
                        padding_length = max_know_length - len(bina_know)
                        bina_know = torch.cat((bina_know, torch.tensor([1] * padding_length, dtype=torch.int64)))
                        know_bin.append(bina_know)
                    knowledge_bin = torch.stack(know_bin).unsqueeze(0)
                    score = roberta.predict('sentence_classification_head', input, return_logits=True, knowledge = knowledge_bin)
                    scores.append(score)
                pred = torch.cat(scores).argmax()
                answer = example['answer_idx']
                # answer = ord(example['answer_idx']) - ord('A')
                nsamples += 1
                if pred == answer:
                    ncorrect += 1
                    corrects.append(example)
    result = {
        'acc': ncorrect / float(nsamples)
    }
    print('Accuracy: ' + str(ncorrect / float(nsamples)))
    output_file = os.path.join(args.model_path, "eval_out.txt")
    with open(output_file, "w") as writer:
        writer.write(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
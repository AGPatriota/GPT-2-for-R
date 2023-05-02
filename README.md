# GPT2 for R


The OpenAI Generative Pre-Trained Transformer [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2#openai-gpt2) is now translated into R language. The original GPT2 paper can be read [here](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and the GPT2 GitHub can be accessed [here](https://github.com/openai/gpt-2). 

The model in this repository is not a wrapper of the original GPT2. It is written entirely in R language with the weights trained by OpenAI. Before running the model, make sure you have downloaded the required weights [here](https://drive.google.com/file/d/1jnYn3kaVyoLcGEmDBCOFNPAslrs2tRo4/view?usp=sharing) to the main folder.

I would like to thank [Álvaro Kothe](https://github.com/Alvaro-Kothe) for helping me to organize some parts of the code in an [early version](https://github.com/AGPatriota/GPT4R).

## Disclaimer

This repository has only academic purposes. The weights are precisely the same ones OpenAI made publicly [available](https://huggingface.co/gpt2/blob/main/tf_model.h5); they only were translated into R language (this can be easily verifiable by inspecting the weight values). That is, I did not train the model with any text or corpora. 

The generated text should not be employed to harm other people, institutions or any other communities and organizations. You are entirely responsible for the use of this tool.

## Dependencies:

- [tok](https://github.com/dfalbel/tok)
- [torch](https://cran.r-project.org/web/packages/torch/index.html) 

## quick start

If you have a GPU, you can try to run the code by specifying the context. First open R in the main folder and type `source('main.R')` to run all the required functions. Then, for each generation we need to call `Generate()` such as: 

```
Generate()
Type your prompt here >>
```

After typing you input and pressing Enter, the system will start to generate new tokens.

## Examples of inputs and outputs

Set the seed to replicate the results:

```
torch_manual_seed(42)
Generate()
```

The input `The discipline of statistics is` produces the following output

```
===================== Generating Tokens =====================

The discipline of statistics is designed to be simple and objective, but can be used to show, e.g., whether a particular statistic relates to the distribution of income divided by the number of people in society at the time it was used. The term "statistical data" is used to indicate that the data collected by the study does not necessarily reflect the facts or the methods used by a study, but is rather a way of depicting the nature and extent of the study. For example, when the data are reported as a
```

If you want to change the maximum number of generated tokens, the temperature or the top_k hyperparameters, specify them in the function Generate() such as

```
Generate(max_new_tokens=10, temperature = 1, top_k = 2)
```

### License

MIT

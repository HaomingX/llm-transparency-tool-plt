<h1>
  <img width="500" alt="LLM Transparency Tool" src="https://github.com/facebookresearch/llm-transparency-tool/assets/1367529/795233be-5ef7-4523-8282-67486cf2e15f">
</h1>
Thanks to interpretability repository provided by Meta.

Our repository retains the functionalities of the original and introduces a new convenient plotting feature using Matplotlib. Additionally, we have modified the code to support direct execution of local checkpoints.

The guide of installation and running are as follows:

## Local Installation


```bash
# download
git clone git@github.com:facebookresearch/llm-transparency-tool.git
cd llm-transparency-tool

# install the necessary packages
conda env create --name llmtt -f env.yaml
# install the `llm_transparency_tool` package
pip install -e .

# now, we need to build the frontend
# don't worry, even `yarn` comes preinstalled by `env.yaml`
cd llm_transparency_tool/components/frontend
yarn install
yarn build
```

## Collect Knowledge Circuits

To collect Knowledge Circuits, execute the run.py script with the following command:

```sh
python run.py --model_path "/path/to/model_ckpt/" --model_name "model-offical-name" --output_dir "/path/to/save.json"
```

Alternatively, you can use the provided shell script:

```bash
bash run.sh
```

## Plot Knowledge Circuits Figures

To generate and plot the Knowledge Circuits figures, run the following command:

```sh
bash plt.sh
```

This will produce visual representations of the Knowledge Circuits based on the collected data.

The effect is as follows:

<img src="https://haoming2003.oss-cn-hangzhou.aliyuncs.com/img/ckpt-0.png" alt="ckpt-0" style="zoom: 15%;" />

## The demo of original github

```bash
streamlit run llm_transparency_tool/server/app.py -- config/local.json
```

### Adding support for your LLM

Initially, the tool allows you to select from just a handful of models. Here are the
options you can try for using your model in the tool, from least to most
effort.

### The model is already supported by TransformerLens

Full list of models is [here](https://github.com/neelnanda-io/TransformerLens/blob/0825c5eb4196e7ad72d28bcf4e615306b3897490/transformer_lens/loading_from_pretrained.py#L18).
In this case, the model can be added to the configuration json file.

### Tuned version of a model supported by TransformerLens

Add the official name of the model to the config along with the location to read the
weights from.

### The model is not supported by TransformerLens

In this case the UI wouldn't know how to create proper hooks for the model. You'd need
to implement your version of [TransparentLlm](./llm_transparency_tool/models/transparent_llm.py#L28) class and alter the
Streamlit app to use your implementation.

## Citation
If you use the LLM Transparency Tool for your research, please consider citing:

```bibtex
@article{tufanov2024lm,
      title={LM Transparency Tool: Interactive Tool for Analyzing Transformer Language Models}, 
      author={Igor Tufanov and Karen Hambardzumyan and Javier Ferrando and Elena Voita},
      year={2024},
      journal={Arxiv},
      url={https://arxiv.org/abs/2404.07004}
}

@article{ferrando2024information,
    title={Information Flow Routes: Automatically Interpreting Language Models at Scale}, 
    author={Javier Ferrando and Elena Voita},
    year={2024},
    journal={Arxiv},
    url={https://arxiv.org/abs/2403.00824}
}
````

## License

This code is made available under a [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license, as found in the LICENSE file.
However you may have other legal obligations that govern your use of other content, such as the terms of service for third-party models.
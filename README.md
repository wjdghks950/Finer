# Finer: A Benchmark for Enhancing Fine-Grained Visual Concept Recognition in Large Vision Language Models


**Finer: Investigating and Enhancing Fine-Grained Visual Concept Recognition in Large Vision Language Models** [[Paper](https://arxiv.org/abs/2310.03744)] <br>
[Jeonghwan Kim](https://wjdghks950.github.io/), [Heng Ji](https://blender.cs.illinois.edu/hengji.html)

## Contents
- [Install](#install)
- [Dataset](#dataset)
- [Evaluation](#evaluation)

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
**Usage and License Notices**: This dataset inherits the Usage and License Notices from LLaVA (Liu et al., 2023). The data and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.


## Install

1. Clone the current repository to reproduce the Finer evaluation
```bash
git clone https://github.com/wjdghks950/Finer.git
cd Finer
```

2. Install and setup the `llava` conda environment from the official [LLaVA repository](https://github.com/haotian-liu/LLaVA) to run LLaVA

3. Install and setup the `lavis` conda environment from the official [LAVIS repository](https://github.com/haotian-liu/LLaVA) for InstructBLIP and BLIP-2 set up

## Dataset
- First, set up a separate `data` directory in the same directory as the `LLaVA` and `LAVIS` dirs.
- Under the `data` directory, set up the following directories separately
    - ddd


## Citation

If you find Finer useful for your research and applications, please cite using this BibTeX:
```bibtex

@misc{kim2024finer,
      title={Finer: Investigating and Enhancing Fine-Grained Visual Concept Recognition in Large Vision Language Models}, 
      author={Kim, Jeonghwan and Ji, Heng},
      publisher={arXiv:2310.03744},
      year={2024},
}
```
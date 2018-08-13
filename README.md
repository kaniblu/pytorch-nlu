# PyTorch Implementation of Natural/Spoken Language Understanding Models #

This is a comprehensive implementation of common joint langauge understanding 
models published in recent years. Joint NLU/SLU models aim to classify 
utterance types and fill-in slots at the same time 
(first proposed by [X. Zhang and H. Wang, 2016](https://www.ijcai.org/Proceedings/16/Papers/425.pdf)). Researches have come up with extensive 
models such as [attention-based RNNs](https://arxiv.org/pdf/1609.01454.pdf) and
[pointer networks](http://www.aclweb.org/anthology/P18-2068) since then. 
By implementing common baselines and recently published works in a single codebase, we intend to make models more easily comparable, thereby facilitating new and 
existing researches in NLU/SLU.
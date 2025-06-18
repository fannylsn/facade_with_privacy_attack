.. image:: https://upload.wikimedia.org/wikipedia/commons/f/f4/Logo_EPFL.svg
   :alt: EPFL logo
   :width: 75px
   :align: right

======
Facade
======

Repository for the artifact of `Fair Decentralized Learning <https://arxiv.org/pdf/2410.02541>`_ published at SaTML 2025.

Decentralized learning (DL) is an emerging approach that enables nodes to collaboratively train a machine learning model without sharing raw data. In many application domains, such as healthcare, this approach faces challenges due to the high level of heterogeneity in the training data's feature space. Such feature heterogeneity lowers model utility and negatively impacts fairness, particularly for nodes with under-represented training data. In this paper, we introduce Facade, a clustering-based DL algorithm specifically designed for fair model training when the training data exhibits several distinct features. The challenge of Facade is to assign nodes to clusters, one for each feature, based on the similarity in the features of their local data, without requiring individual nodes to know apriori which cluster they belong to. Facade (1) dynamically assigns nodes to their appropriate clusters over time, and (2) enables nodes to collaboratively train a specialized model for each cluster in a fully decentralized manner. We theoretically prove the convergence of Facade, implement our algorithm, and compare it against three state-of-the-art baselines. Our experimental results on three datasets demonstrate the superiority of our approach in terms of model accuracy and fairness compared to all three competitors. Compared to the best-performing baseline, Facade on the CIFAR-10 dataset also reduces communication costs by 32.3% to reach a target accuracy when cluster sizes are imbalanced.


------
Citing
------

Cite us as ::

   @inproceedings{biswas2024fair,
     title={{Fair Decentralized Learning}},
     author={Biswas, Sayan and Kermarrec, Anne-Marie and Sharma, Rishi and Trinca, Thibaud and de Vos, Martijn},
     booktitle={3rd IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)},
     year={2025},
     url={https://arxiv.org/abs/2410.02541}
   }



-------
License
-------

This project is licensed under the MIT License. See the LICENSE file for more details.

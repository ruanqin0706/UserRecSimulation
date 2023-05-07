# UserRec

This repository contains the source code for the paper titled "Unveiling the Relationship between News Recommendation Algorithms and Media Bias: A Simulation-based Analysis of the Evolution of Bias Prevalence", 
which aims to direct investigate the relationship between news recommendation algorithms and media bias. 
We designed a news recommendation simulation framework to evaluate the impact of media bias on different recommendation algorithms under different user choice strategies. 
The project comprises four folders: Articles, Users, Recommenders, and Observers.

### Articles

We provide the news dataset modified on the [SemEval-2019 Task 4 Hyperpartisan Dataset](https://pan.webis.de/semeval19/semeval19-web/) under the articles directory. The datasets used in the experiments are located in the `articles/processed_data` directory.

### Users

The code for generating synthetic users is included under the user's directory. The user groups tested in the experiments are in the `users/synthetic_user_groups` directory.

### Recommenders

The NAML, NPA, and NRMS algorithms use the official implementation of [Microsoft Recommenders](https://github.com/microsoft/recommenders). The FIM and PLM-empowered algorithms were re-implemented using the [PyTorch framework](https://github.com/pytorch/pytorch).

### Observers

This section consists of simulation, scripts, analyzer, and results. The simulation folder implements the feedback loop interaction between users, news recommendation algorithms, and candidate news sets. 
The user choice strategies reported in the paper are implemented in the `observer/simulation/simulate_feedback.py` file.
The scripts folder helps to connect the observation process.
The analyzer folder analyzes the results of the evolution of bias prevalence in users' browsing histories, and the analyzed results are recorded in the result folder.


Feel free to contact us if you have any questions or need support.:)

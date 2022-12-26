In this folder, we provide a few notebooks containing the experiments related to CBO modifications. Notebooks descriptions:
| **name** | **description** |
| --- | --- |
| 01-partial-vs-full-update.ipynb | Comparison of dynamics with full and partial updates. The partial update implies that after processing a particles-level batch, the corresponding update will only be applied to particles of that batch. During the full update, all the particles are updated after processing each batch. |
| 02-graidents-usage.ipynb | Comparison of dynamics with and without using additional gradient term. |
| 03-additional-random-shift.ipynb | Comparison of dynamics with different values of the threshold for additional random drift (epsilon). |
| 04-lenet5.ipynb | Experiment with training the LeNet5 architecture (with approximately 40K trainable parameters) on MNIST. |

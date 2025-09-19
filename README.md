Example of using docker to run task using tensorglow v1.

The code will execute the network model from the paper:

<a href="https://arxiv.org/abs/2208.01284">[In-Hand Pose Estimation and Pin Inspection for Insertion of Through-Hole Components]</a>

And color the cad models in the folder cad/


Docker commands:

```bash
docker build -t pin .
```

```bash
docker run -v /home/frhag/workspace/docker_pin/cad:/app/data --rm -it pin
```

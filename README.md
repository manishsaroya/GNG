
# Roadmap-Learning-with-Topology-Informed-Growing-Neural-Gas
**Contact: saroyam@oregonstate.edu**

## About
This repository generates navigation roadmaps from probabilistic occupancy maps of uncertain and cluttered environments. 

## Paper
```
@article{saroya2021roadmap,
  title={Roadmap Learning for Probabilistic Occupancy Maps with Topology-Informed Growing Neural Gas},
  author={Saroya, Manish and Best, Graeme and Hollinger, Geoffrey A},
  journal={IEEE Robotics and Automation Letters},
  volume={6},
  number={3},
  pages={4805--4812},
  year={2021},
  publisher={IEEE}
}
```
**Abstract**
We address the problem of generating navigation roadmaps for uncertain and cluttered environments represented with probabilistic occupancy maps. A key challenge is to generate roadmaps that provide connectivity through tight passages and paths around uncertain obstacles. We propose the topology-informed growing neural gas algorithm that leverages estimates of probabilistic topological structures computed using persistent homology theory. These topological structure estimates inform the random sampling distribution to focus the roadmap learning on challenging regions of the environment that have not yet been learned correctly. We present experiments for three real-world indoor point-cloud datasets represented as Hilbert maps. Our method outperforms baseline methods in terms of graph connectivity, path solution quality, and search efficiency. Compared to a much denser PRM*, our method achieves similar performance while enabling a 27× faster query time for shortest-path searches.

## Requirements
- [Gudhi](https://gudhi.inria.fr/python/latest/installation.html)
- [Neupy](http://neupy.com/pages/installation.html)

## Usage
```
python persistence/gng_neupy_run.py
```

## Roadmap
   ![](https://github.com/manishsaroya/GNG/blob/master/gng.gif)

# Run Tensorflow in a Docker container with CUDA support on AWS

This repository contains all code to run a Keras app using Tensorflow in a Docker container with CUDA support on AWS.

## One does not simply pip install... the hassle of configuring environments

In my years as a Data Scientist there is one mundane task that keeps returning with each new project, i.e. setting up the
environment in which we will run our code. Fair enough, setting up a new environment has increased in 
simplicity significantly over the years. Tools such as Anaconda, Jupyter, pipenv, PyCharm have taken away annoying tasks 
such as installing the proper Python version, setting the correct interpreter paths and installing the proper version of 
all dependencies. Additionally, having proper conventions in place such as a `README` and a `requirements.txt` file helped 
reduce errors even more. 

However, they still required some manual steps, such as installing binaries, which are prone to errors. Especially when transferring
work to other colleagues, who might be running a slightly differently configured OS, time-consuming environment configuration 
errors would start to happen.

![Source: https://xkcd.com/1987/](https://imgs.xkcd.com/comics/python_environment.png)


## It's Docker time!

So how did we manage to spend less time on configuring new environments, without doing any concessions on the quality 
of the environments? Say hello to **Docker**! Docker images and containers give us pre-configured environments that are 
ready to start with. They also guarantee us that person A and B have identical environments in which the code is run, which will improve the
stability and reliability of the code as well. The last thing we want is that person A gets different output from the 
code than person B, where identical output is expected. Or that person B is not even able to reproduce your results because the 
code won't even run due to missing dependencies. Finally, if the code is already containerized, we can easily deploy and 
scale our code using services such as Kubernetes. 
 
## It's not over till the fat lady sings...

By now you probably can't wait to start using Docker and run all your Deep Learning models as Docker containers. However, 
we first need to create a Docker image that is going to give us a pre-configured environment with all Deep Learning packages 
installed. That is where things can get a bit more complicated though...

One option is by browsing through the list of Deep Learning images within the public
[Docker Hub repository](https://hub.docker.com/search/?isAutomated=0&isOfficial=0&page=1&pullCount=0&q=deep+learning&starCount=0).
Here you will find a list of publicly available Docker images that will provide you with several pre-configured Deep Learning images. 

Although

**... TO BE CONTINUED ...**


## Creating a Docker image with CUDA support

**... TO BE CONTINUED ...**


## Scale up by running the app in a Docker container on AWS

Using our newly creating Docker image with CUDA support, we want to move

1. Start an AWS EC2 instance with Ubuntu 16.04. The current code is designed to run on a p2.xlarge (spot) instance.

2.  Run the commands in `scripts/ec2_instance_setup.sh` to install all 
NVIDIA, CUDA and Docker dependencies. This can done either manually by SSHing into the instance, or by pasting the 
code of the script in the UserData field when starting an EC2 instance.

3. SSH into the instance and clone this repository using Git onto the instance using 
`git clone https://github.com/thomhopmans/themarketingtechnologist.git`

4. Browse to this directory, i.e. `themarketingtechnologist/10_docker_tensorflow_with_cuda_on_aws`.

5. Build a Docker image of the app using `docker build -t cuda_app .`

6. Run the app in a Docker container using `docker run --runtime=nvidia --rm -d --shm-size=1g cuda_app`. 

7. Follow the logs using `docker logs CONTAINER_NAME -f`.

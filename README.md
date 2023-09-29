# Consumer-Complaint-Dispute-Prediction

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
- [![Git Logo](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.flaticon.com%2Ffree-icon%2Fgithub-logo_25231&psig=AOvVaw0PLSY7LmQTWcSX5I9EKBW4&ust=1696114076785000&source=images&cd=vfe&ved=0CBEQjRxqFwoTCNiCgrvz0IEDFQAAAAAdAAAAABAE)](https://git-scm.com/)
- [![Python Logo](https://example.com/python-logo.png)](https://www.python.org/downloads/)
- [![Docker Logo](https://example.com/docker-logo.png)](https://www.docker.com/get-started)



## Problem Statement

Design and construct a scalable machine learning pipeline capable of predicting whether a given consumer complaint will result in a dispute or not.

## Solution Proposed

The proposed solution for this project involves the development of a robust and scalable machine learning pipeline. This pipeline will encompass data collection, preprocessing, feature engineering, model training, and evaluation. Leveraging historical consumer complaints data, the machine learning models will be trained to predict whether a given complaint is likely to result in a dispute. By employing state-of-the-art algorithms and techniques, we aim to provide accurate and actionable insights for financial institutions, helping them proactively address consumer concerns and improve their dispute resolution processes. This solution holds the potential to enhance customer satisfaction, reduce dispute-related costs, and ultimately contribute to a more efficient and responsive financial services industry.



## Technologies Used:
- Python
- PySpark
- PySpark ML
- Airflow as Scheduler
- MongoDB

## How to run?


1. Clone the repository to your local machine using the following command:

   ```bash
   git clone https://github.com/SuyodhanJ6/Consumer-Complaint-Dispute-Prediction.git

2. Navigate to the project directory using the `cd` command:

   ```bash
   cd Consumer-Complaint-Dispute-Prediction

3. Run the initialization script to set up the project:

   ```bash
   bash init_setup.sh

4. Finally, you can run the Python application:

   ```bash
   python app.py

### Setting Up Environment Variables for local run 

You can create a `.env` file in the project directory and add these variables with their respective values. Here's an example of how your `.env` file should look:

   
      AWS_ACCESS_KEY_ID=your-aws-access-key
      AWS_SECRET_ACCESS_KEY=your-aws-secret-key
      MONGO_DB_URL=your-mongodb-url
      ECR_REPOSITORY_NAME=your-ecr-repository-name
      ECR_REPOSITORY_URL=your-ecr-repository-url

## Infrastructure Required:
- Amazon Web Services (AWS) EC2 Instances
- Amazon S3 Bucke
- Google Cloud Artifact Registry

## WorkFLow setup

Before running this application, you need to set the following environment variables in github:

1. **AWS_ACCESS_KEY_ID**: Your AWS access key.
2. **AWS_SECRET_ACCESS_KEY**: Your AWS secret access key.
3. **MONGO_DB_URL**: URL of your MongoDB instance.
4. **ECR_REPOSITORY_NAME**: Name of your Amazon ECR repository.
5. **ECR_REPOSITORY_URL**: URL of your Amazon ECR repository.

### Creating an EC2 Instance

To deploy and run this project, you'll need an Amazon EC2 instance. Follow these steps to create one:

1. **Log in to your AWS Console**: Visit the [AWS Management Console](https://aws.amazon.com/console/) and log in to your AWS account.

2. **Launch EC2 Instance**:
   - In the AWS Management Console, navigate to the EC2 dashboard.
   - Click the "Launch Instance" button to start the EC2 instance creation process.
   - Choose an Amazon Machine Image (AMI) that suits your needs. A Linux-based AMI (e.g., Amazon Linux 2) is recommended.
   - Select an instance type based on your project's requirements.
   - Configure instance details, such as network settings and storage.
   - Add any required tags and configure security groups and key pairs.
   - Review your settings and launch the instance.

3. **Access EC2 Instance**:
   - Once your instance is running, you can access it via SSH. Use the private key associated with the key pair you selected during instance creation.

### Creating an Amazon ECR Repository

To host Docker images for this project, you'll need an Amazon Elastic Container Registry (ECR) repository. Follow these steps to create one:

1. **Log in to your AWS Console**: Visit the [AWS Management Console](https://aws.amazon.com/console/) and log in to your AWS account.

2. **Navigate to ECR**:
   - In the AWS Management Console, navigate to the Amazon ECR service.

3. **Create a Repository**:
   - Click the "Create repository" button.
   - Provide a name for your repository, e.g., "consumer-complaint-ecr."
   - Optionally, configure repository settings and permissions.
   - Click the "Create repository" button to create the ECR repository.

4. **Access Repository Information**:
   - After creating the repository, you can access its details, including the repository URI, which you'll need for Docker image tagging and pushing.

That's it! You now have an EC2 instance to deploy your project and an ECR repository to host Docker images.


### Creating an Amazon S3 Bucket

![S3 Bucket](https://drive.google.com/file/d/1epQI13n79SLXNgNZXGSKjMkMe6E17Qem/view?usp=sharing)

To store and manage project-related data, you can create an Amazon S3 bucket. Follow these steps to create one:

1. **Log in to your AWS Console**: Visit the [AWS Management Console](https://aws.amazon.com/console/) and log in to your AWS account.

2. **Navigate to S3**:
   - In the AWS Management Console, navigate to the Amazon S3 service.

3. **Create a Bucket**:
   - Click the "Create bucket" button.
   - Provide a unique name for your bucket, following AWS naming guidelines.
   - Select the AWS region where you want to create the bucket.
   - Configure additional settings, such as versioning, logging, and access control.
   - Review your settings and click the "Create bucket" button.

4. **Access Bucket Information**:
   - After creating the bucket, you can access its details, including the bucket name and the unique bucket URL.

### Using the S3 Bucket

Once your S3 bucket is created, you can use it to store and manage project-related files and data. You can also configure access permissions and integrate it with your project as needed.

Remember to secure your S3 bucket and configure access policies to control who can access and modify its contents.

That's it! You now have an S3 bucket for storing and managing your project's data.

## Project Setup Guide

### Step 1: Launch an EC2 Instance

1.1. **Create an EC2 Instance:** Use the AWS Management Console to create a new EC2 instance. Select an appropriate instance type and security group settings for your project.

1.2. **SSH into the EC2 Instance:** After launching the EC2 instance.

### Step 2: Prepare the EC2 Instance
2.1. **Update and Upgrade Packages:** Once connected to the EC2 instance, update the package list and upgrade installed packages:
```bash
sudo apt-get update -y
sudo apt-get upgrade -y
```
2.2. **Install Docker:** Install Docker on the EC2 instance by running the following commands:
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

```
2.3. **Add User to Docker Group:** To run Docker commands without using sudo, add your user (in this case, ubuntu) to the docker group:
```bash
sudo usermod -aG docker ubuntu
```
2.4. **Activate the Docker Group Changes:** To apply the group changes immediately, use the newgrp command:
```bash
newgrp docker
```

### Setting Up GitHub Actions Runners

To use GitHub Actions runners for your project, follow these steps:

1. **Open GitHub Actions Runner:** Launch a GitHub Actions runner on your EC2 instance by following the GitHub Actions documentation.

2. **Modify Runner Configuration:** Customize the runner configuration according to your project's requirements.

3. **Follow GitHub Actions Instructions:** In your GitHub repository, navigate to the Actions tab and follow the instructions to set up and configure the runner.

4. **Run the Initialization Script:** Finally, on your EC2 instance, navigate to the project directory and run the initialization script:

   ```bash
   ./run_init.sh
   ```

## License
- This script is open-source and available under the MIT License. You are free to use and modify it for your projects.

## Contribution
- Contributions are welcome! If you have ideas for improvements or new features, please open an issue or submit a pull request.

## Contact
- For questions or feedback, feel free to reach out to the project maintainers at prashantmalge181@gmail.com or create an issue on GitHub.

Enjoy streamlining your project setup process with the Project Template Generator Script!
# Seriousness Prospects:<br />Using a chatbot to identify serious candiates
This is a consulting project that I work on in Insight Data Science @ NY 2019C. I collaborate with a lead data engineer from a startup in India as a data science consultant. 

## Motivation
In India, food delivery market is rapidly increasing recently. The revenue is expected to be over more than 5 billion US dollars in 2023. As all of the food delivery companies need a large amount of delivery employees, the recruitment process will be incredibly dreadful. 

Vahan, the company I work with, has improved the screening process by developing a scalable and automated chatbot on What's App. This product can not only engage with job applicants and collect their information through a conversation, but also begin a real-time screening process and schedule interviews for them. 

However, they notice a new business problem happened after the screening process. Only less than 4% job applicants who received interview opportunities actually proceeded on-site interview. Therefore, my task is to establish a prediction model identifying serious applicants based on messages between the chatbot and job applicants.  

## Tech stack
1. Python3
2. Pandas
3. Im-balanced learn: Resamling tools
4. GitHub: Version control
5. Flask: Demo

## Files
1. [Final] Unsuprvising learning.ipynb: <br />
Dataset: User information + features from message data (10,535 datapoints) <br />
Analysis: Visualize resampling-data, compare resuls from different models
2. PCA_original10534_additional258.ipynb: <br />
Dataset: User information + features from message data (10,535 + additional 120 datapoints) <br />
Analysis: Visualize data
3. Basline modeling.ipynb, Merge_modeling.ipynb: <br />
Compare user information only to user information + features from message data in modeling <br />
The comparision indcludes different resampling tools.
# Job Matching Enhancement using Dynamic Recommendations

## Live Demo

You can test the live demo of our job matching enhancement system using the following link:

[Link to Demo Test](http://54.82.18.196:5000/)

**Note:** The API is hosted on an AWS EC2 instance, so it may take a few seconds to respond.

**Pitch:** https://drive.google.com/file/d/17Oa2dHjgnuJ22qGp-8Wljsx2_EM52q3p/view?usp=sharing

## Introduction to the Problem and Context
Finding job opportunities that align with candidates' profiles is a significant challenge. Traditional systems rely heavily on keyword matching and explicit job descriptions, which limits the discovery of relevant job openings. This creates inefficiencies and missed opportunities in the job market. Enhancing the matching process can connect candidates with new and unexpected opportunities, improving both individual career growth and market efficiency.

## Review of Existing Solutions

The study by Alonso et al. (2023) presents a recommendation architecture that integrates NLP and machine learning modules to analyze and classify candidate profiles based on the skills required for different positions. The model employs specific components: an interface for information collection, a database to store occupation descriptors, a text extractor that identifies relevant terms in resumes, a transformer-based module to convert text into vectors, and a scoring module to rank candidates based on the similarity between their profiles and job openings. However, this approach faces limitations due to its reliance on structured and predefined descriptors, restricting its application in domains that demand greater flexibility. My solution addresses this limitation by enabling more dynamic and adaptable recommendations.

## Proposed Solution
The solution consists of a frontend where users provide their resumes and interests. This information is sent to a prompt that uses the GPT API to generate career suggestions based on the user’s profile and preferences. From this response, both the suggested careers and the user’s resume are represented in a vector space, with greater weight assigned to the user’s interests. This representation allows for comparison and evaluation of the alignment between the user’s profile and the available job openings in the database. As a result, we can recommend opportunities that, while not immediately obvious, are highly aligned with the user’s profile and goals.

## Results and Impact Metrics

Our results were mensured testing the solution with people and getting feedback from them. The feedback was positive, and the users were able to find job opportunities that they would not have found otherwise. Videos and testimonials below:

- [Drive with the feedbacks](https://drive.google.com/drive/folders/1g3lcXIaqWC_K5QNyEq7VCAdRrG5MwpyL?usp=sharing)

- [Breno Quessie - Video](https://drive.google.com/file/d/1BlzPo7iqxsY5GG2iwZeKbRKUZNTFxwDo/view?usp=sharing)

- [Bruno Sanches - Video](https://drive.google.com/file/d/1JRHsEH5HhpO2UKVX1ETTnjvi45w3DD1h/view?usp=drive_link)

- [Rafael - Video](https://drive.google.com/file/d/1LhZdm0ofP_RqvrrrKxN0KFb0NQqQeZCm/view?usp=sharing)

- [Sarah Pimenta - Video](https://drive.google.com/file/d/1BLEnBxTncbwb5g0GMDO3aT-PZVIGk6d2/view?usp=sharing)

- [Ellen - Video](https://drive.google.com/file/d/1kaymb4ZgCnmnQInh-O_g-tccPRPf_BwH/view?usp=sharing)

## Functionalities of the Application

   - **Web server** 
      - **Input:** Client's resume and interests
      - **Output:** List of all job openings that match the client's profile 


## Next Steps
- Expand the number of partner companies to diversify the job types in the database. This will enable the platform to serve a broader audience and provide opportunities across various industries and career levels.

- Accept PDF files as input to enhance user convenience and accessibility.

- Implement a feedback system to refine recommendations based on user interactions and preferences.

- Integrate a chatbot feature to provide real-time assistance and guidance to users during the job search process.

## References
Alonso, R., Dessí, D., Meloni, A., & Recupero, D. R. (2023). *A General and NLP-based Architecture to Perform Recommendation: A Use Case for Online Job Search and Skills Acquisition*.

## Run the project locally

   To run the project locally, follow these steps:

   1. Clone the repository:
   ```
   bash
   git clone https://github.com/sarahp31/aps-2-nlp
   ```

   2. Give permissions to files:
   
   ```
   bash
   chmod +777 setup_git_hook.sh
   chmod +777 run_pipeline.sh
   ```

   3. Run the setup script:
   ```
   bash
   ./setup_git_hook.sh
   ```

   4. Give data permissions to the run script:
   ```
   bash
   chmod 664 data/careers.db
   chmod 775 data/
   ```

   5. Run the project pipeline installation, scrapping, embedding, and training:
   ```
   bash
   ./run_pipeline.sh
   ```

   6. Run the API:
   ```
   bash
   python matching_jobs_api/main.py
   ```

   7. Access the API at ```http://localhost:5000```


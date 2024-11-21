# Job Matching Enhancement using Dynamic Recommendations

Link para o site: http://54.82.18.196:5000/

## Introduction to the Problem and Context
Finding job opportunities that align with candidates' profiles is a significant challenge. Traditional systems rely heavily on keyword matching and explicit job descriptions, which limits the discovery of relevant job openings. This creates inefficiencies and missed opportunities in the job market. Enhancing the matching process can connect candidates with new and unexpected opportunities, improving both individual career growth and market efficiency.

## Review of Existing Solutions
A study by **Alonso et al. (2023)** presents a recommendation architecture integrating NLP and machine learning modules to analyze and classify candidate profiles based on required skills for various positions. Their model includes:
- An **interface** for collecting candidate information.
- A **database** for storing occupation descriptors.
- A **text extractor** to identify relevant terms in resumes.
- A **transformer-based module** to convert text into vectors.
- A **scoring module** to rank candidates by profile-job similarity.

However, this approach depends on structured and predefined descriptors, limiting its flexibility in dynamic and complex domains. 

### Proposed Solution's Advantages
Our proposed solution addresses these limitations by enabling **dynamic and adaptable recommendations**, creating a more flexible and personalized job matching process.

## Proposed Solution
This solution includes:
1. **Frontend Interface**: 
   - Users submit their resumes and specify career interests.
   
2. **Prompt-based Suggestions**:
   - User input is sent to a GPT-based API, which generates career suggestions tailored to the profile and interests.

3. **Vector Space Representation**:
   - Both the user's resume and the suggested careers are represented in a **vector space**, with higher weights assigned to user interests.

4. **Comparison and Evaluation**:
   - Profiles and job openings in the database are compared using this representation to evaluate alignment.

### Key Benefits
- Recommends **non-obvious opportunities** aligned with the user’s profile.
- Incorporates user preferences dynamically.
- Improves the discoverability of new and relevant job opportunities.

## Results and Impact Metrics
The application demonstrates:
- **Quantitative Results**: (Insert metrics such as improved job match rates, reduced search time, etc.)
- **Qualitative Feedback**: (Include user testimonials highlighting the value and relevance of recommendations.)

These results confirm the solution's ability to effectively meet user needs and improve job search experiences.

## Next Steps
Expand the number of partner companies to diversify the job types in the database. This will enable the platform to cater to a broader audience and offer opportunities across various industries and career levels.

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


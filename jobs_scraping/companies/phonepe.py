import requests

from base.apijobscrapper import APIJobScrapper


class PhonePeApiJobScrapper(APIJobScrapper):

    def get_json_data(self, url):
        return requests.get(url).json()['jobs']

    def get_name(self):
        return "PhonePe"

    def get_image_url(self):
        return "data:image/png;base64,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"

    def get_jobs(self):
        url = 'https://boards-api.greenhouse.io/v1/boards/phonepe/jobs'
        data = self.get_json_data(url)
        postings = []
        for doc in data:
            url_job = 'https://boards-api.greenhouse.io/v1/boards/phonepe/jobs/' + str(doc['id'])
            doc = requests.get(url_job).json()
            postings.append({})
            postings[-1]['title'] = doc['title']
            postings[-1]['description'] = doc['content']
            postings[-1]['url'] = doc['absolute_url']
            postings[-1]['location'] = doc['location']['name']
        return postings

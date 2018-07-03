import numpy as np 
from lightfm.datasets import fetch_movielens #Our fetching method
from lightfm import LightFM #Our Model Class

#Fetch data
data = fetch_movielens(min_rating=5.0)

#Create Model 
model = LightFM(loss='warp') #WARP (Weighted Approximate-Rank Pairwise)
model.fit(data['train'], epochs=30, num_threads=4)

#Sample Recommendation
def sample_recommendation(model, data, user_ids):

	#Numbers of users and movies in training data
	n_users, n_items = data['train'].shape

	#Generate recommendations for each user
	for user_id in user_ids:
		
		#Movies they already like
		know_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

		#Movies our model predicts
		scores = model.predict(user_id, np.arange(n_items))

		#Put them in order of most liked
		top_items = data['item_labels'][np.argsort(-scores)]

		#Print out the results 
		print("User %s" % user_id)
		
		print("User liked movies:")
		for x in know_positives[:3]:
			print("                     %s" % x)

		print("Recommendation:")
		for x in top_items[:3]:
			print("                     %s" % x)

sample_recommendation(model, data, [3, 25, 450])
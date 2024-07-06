# Result Analysis:
Positive Sentiment (58.41%):
The majority of the customer reviews express positive sentiments. This indicates that most customers are satisfied with their experience at the restaurant. Positive feedback could relate to various aspects such as food quality, service, ambiance, or value for money. We shall do detail analysis with category-wise polarity

Negative Sentiment (22.03%):
A significant portion of reviews is negative. This suggests there are notable areas of concern that need attention. We shall nitpick the issues in the category-wise analysis.

Neutral Sentiment (17.75%):
A substantial number of reviews are neutral. These reviews likely contain mixed feelings or mention aspects that did not strongly impact the customer's overall experience.


# Category-Wise Polarity Analysis
1. Ambience:
Most of the reviews (69%) are positive, indicating customers enjoy the restaurant's atmosphere.
A significant portion (21%) of reviews are negative, suggesting there are some issues with the restaurant's environment that need addressing. We can see the top 10 negative aspects of ambience.

2. Food
The majority of food-related reviews are positive (66%), indicating that customers are generally satisfied with the food quality.
Some negative(17%) feedback exists, pointing to areas where the food could be improved. We could observe that some customers are not satisfied with the portions.
A significant portion of neutral(14%) reviews, suggesting that while the food is satisfactory, it may not be outstanding. We could observe that some customers are not satisfied with the menu or food options.

3. Price
Most reviews (59%) regarding price are positive, indicating customers feel the pricing is fair or good value.
A significant portion of negative (16%) feedback suggests some customers find the prices too high or not worth it.
A fair amount (14%) of neutral sentiment indicates mixed feelings about the pricing.

4. Service
Less than half of the service-related reviews are positive, indicating some dissatisfaction with the service..
A high percentage of negative (33%) feedback suggests significant issues with the service that need urgent attention..
A substantial amount (19%) of neutral sentiment indicates mixed or indifferent experiences with the service.

# Actionable Insights
Service: The restaurant needs to focus on the service aspect, as it can be observed that from the category wise analysis, most of the customers are dissatisfied with their service. Areas of improvements are,
  # Reduce longer wait time. Following are the comments.
      The service was a bit slow, but they were very friendly.
      The service was bad, the food took to forever to come, we sat on the upper level.
      You have to increase the service a lot.
      Everything, from the soft bread, soggy salad, and 50 minute wait time, with an incredibly rude service to deliver below average food.
  # Staff to attend customer with patience and cordial behaviour.
      The service does sometimes lack focus and it is not ideal if you are in a hurry but I have never been treated rudely.
      The staff is no nonsense.
      The staff ignored my friends and I the entire time we were there.
      If you don't mind pre-sliced low quality fish, unfriendly staff and a sushi chef that looks like he is miserable then this is your place.

Food: Some customers are not satisfied with the food that is being served. There are things the restaurant need to focus on.
  # Increase the portions of food served.
    The food was pretty good, but a little flavorless and the portions very small, including dessert.
    After we got our sashimi order, I could not believe how small the portions were!
    The food was lousy - too sweet or too salty and the portions tiny.
    The portions are small but being that the food was so good makes up for that.
  # Improve on the quality of Food served.
    Quite frankly, this is some of the worst sushi I have ever tried.
    The sushi was awful!
    the food was undercooked-the sauce watery, and the vegetables raw.
    The fish was not fresh and the rice tasted old and stale.
  # Include more items in the menu.
    The menu is very limited - i think we counted 4 or 5 entrees.
    The menu is limited but almost all of the dishes are excellent.
    The menu is nothing like the one on the website.
    This wasn't the expected menu comprised only of pad thai and tom yum soup, but I thought that was what made the place so special.

# Report

Pros:
  - Modular
    Building three separate models (Aspect Entity Extraction, Aspect Polarity Detection, Aspect Category Detection) allows for modularity and easier debugging and maintenance.
    Each model can be independently improved or replaced without affecting the others.
  - Fine-tuning Pretrained Models
    Utilizing pretrained models like distilbert-NER and emotion-english-distilroberta-base leverages transfer learning, which led to better performance even with limited training data.
Cons:
  - Data Preprocessing:
    I did not spend time on text preprocessing or data cleaning, I think we could improve the performance, if the data preprocessing and cleaning is performed on the data.
    I did not address the comments which include sarcasm. 
  - Aspect Term and Category Context:
    By combining the aspect term with the sentence for polarity and category detection, the model might get confused with repetitive or irrelevant context. This is observed in the polarity detection model performance, the polarity model MCC is 0.6 which is less.
    Separate context handling or more sophisticated context-aware training might give better performance.
  - Flat Approach Limitations:
    Treating each task independently we may miss out on potential correlations between aspect extraction, polarity detection, and category detection.


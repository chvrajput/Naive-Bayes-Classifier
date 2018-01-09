# Naive-Bayes-Classifier in Python
Naive Bayes Classifier on text data. It is an example of multiclass classification and on a real scenerio. This is data of ivr while playing when we are not ableto receive the call or number is switchedoff ,unavailable and Busy etc. Mostly cases I am covered in it and this is very interesting and I am able to predict the category on upcoming ivr/audio/recording.

To understand the naive Bayes classifier we need to understand the Bayes theorem. So let’s first discuss the Bayes Theorem.
What is Bayes Theorem?
Bayes theorem named after Rev. Thomas Bayes. It works on conditional probability. Conditional probability is the probability that something will happen, given that something else has already occurred. Using the conditional probability, we can calculate the probability of an event using its prior knowledge.

Below is the formula for calculating the conditional probability.

{P(H \ E) =  { P(E \ H) * P(H)}/ P(E)
where

    P(H) is the probability of hypothesis H being true. This is known as the prior probability.
    P(E) is the probability of the evidence(regardless of the hypothesis).
    P(E|H) is the probability of the evidence given that hypothesis is true.
    P(H|E) is the probability of the hypothesis given that the evidence is there.

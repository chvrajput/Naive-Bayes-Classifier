from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier

newsTrainer = Trainer(tokenizer)

# You need to train the system passing each text one by one to the trainer module.
newsSet =[
    {'text': 'sorry sir mobile you are calling is watstaph means call back in some time aur sanderson at by dynasty followed by the number', 'category': 'switchoff'},
    {'text': 'the number you are calling is others which off or not reachable at the moment please try later aapke dwara dial kiya gaya number ya abhi switched off hai ya network kshetra se bahar hai kripya', 'category': 'switchedoff'},
    {'text': 'jis grahak ko aap call kar rahe hain woh is samay available nahi hai ab aap apne karan tariff ke hisaab se voice message chod sakte hain voice message  ke liye star ek dabaye the customer you are calling is on', 'category': 'unavailable'},
    {'text': 'her lines to the calls destination appised call again like call ki ek he number ke liye abhi bhi nine hi guest hai kripya dobara call karenge sir lines to the calls destination up', 'category': 'Busy'},
    {'text': 'the number you are calling is either switched off or not reachable at the moment please try later aapke dwara dial kiya gaya number ya abhi switched off hai ya network kshetra se bahar hai kripya', 'category': 'switchedoff'},
    {'text': 'the airtel subscriber you have called is speaking to someone else you can wait', 'category': 'Busy'},
{'text':'aapke dwara dial kiya gaya number which stop hai kripya kuch bhi bahut by karen ek hi honi hai payenge aatharah mein parampara chaar karimpuri abhay tratsky rt two in','category':'switchedoff'},
    {'text':'the airtel subscriber you have called is speaking to someone else the person you are calling is  busy please call again later aap jis vyakti se sampark karna chahte hain wo abhi vyast hai kripya thodi der','category':'Busy'},
    {'text':'the number you are calling is either switched off or not reachable after moment please try later aapke dwara dial kiya gaya number ya abhi','category':'switchedoff'},
    {'text':'job number se upper call karta hath to satyapan rahi kripya thoda bhayanak keram sorry the number you are calling is  switched off please call again later','category':'switchedoff'},
    {'text':'so kijiye jitna hai baar sir aap call kar rahe hai sir to isliye banaya tha hum bahut baad call karein ya chaah paida kar raha hai to dial ka hai or wo patniji seven to wo','category':'Busy'},
    {'text':'the number you are calling is either switched off or not reachable at the moment please try later aapke dwara dial kiya gaya number ya abhi switched off hai ya network kshetra se bahar hai kripya','category':'switchedoff'},
    {'text':'the number you have dialled could not become please check number dial kiya gaya number maujood nahi hai kripya number check kar raha hoon','category':'Invalid'},
    {'text':'the person you are calling is  busy call again later aap jis vyakti se sampark karna chahte hain wo abhi vyast hai kripya thodi der baad call karen','category':'Busy'},
    {'text':'the airtel subscriber you have called is speaking to someone else you can wait or call again later aap jis airtel subscriber ko call kiya hai woh abhi dusri call pe vyast hai kripya pratiksha karein ya kuch','category':'Waiting'},
    {'text':'this call cannot be completed at this moment please try again later this call cannot be completed at this moment please try again later','category':'Cannot be Completed'},
{'text':'the number you have dialled could not to count check number dial kiya gaya number maujood nahi hai kripya number check kar raha hoon','category':'Invalid'}
]
for news in newsSet:
    newsTrainer.train(news['text'], news['category'])

# When you have sufficient trained data, you are almost done and can start to use
# a classifier.
newsClassifier = Classifier(newsTrainer.data, tokenizer)

# Now you have a classifier which can give a try to classifiy text of news whose
# category is unknown, yet.
classification = newsClassifier.classify("please check the number you have dial dial kiya hua number kripya jaanch")

# the classification variable holds the detected categories sorted
print(classification)


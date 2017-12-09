# Soper_CS_5984_project  

**Robert Soper, pid: rsoper**  
10 Dec 2017

## "Automating Turing’s Test: An Encoder-Decoder for Chatbot Identification"

*This work is the opinion of the author in partial fullfillment of Viriginia Tech class CS 5984 Deep Learning and does not reflect the views of the U.S. Government, the Department of Defense, or the Defense Intelligence Agency.*

**Summary rationale for this project:**  
- Chatbots can have applications for automation and efficiency in business, but increasing they are having a negative impact on the public discourse in the social media domain as evidenced by Senate hearings and academic studies of influence on the last U.S. election cycle and FCC position on net-neutrality vote "bot or not"
- Researchers at Indiana University and elsewhere have shown that traditional machine learning approaches based on a broad range of features can be effective at discriminating chatbot entities from human entities on Twitter and other social media platforms
- However, we observe that features currently most useful to traditional discrimination of chatbot from humans are the most easy to obsfucate.  Over time there may be increased effort to hide bot activity to protect their effectiveness.  More difficult will be the ability of chatbots to minimic human dialog with high accuracy, although recent advances in deep learning technique will likely continue to improve this performance.  In the future, it will likely be required to include sophisticated evaluation of entity dialog characteristics as part of bot discrimination.
- Identifying a chatbot by analyzing the way in which it responds in a communication resembles the classic Turing test for sentience of an artificial intelligence, hence the title of this project. 

**Summary of technical approach:**
- Lacking a large corpus of social media dialog data and associated metadata, this project uses the Cornell Movie Dialog corpus as a proxity to build scoped training and test datasets.
- A simulated chatbot response was constructed using a LSTM recurrent-neural-network (RNN) encoder-decoder sequence-to-sequence (seq2seq) model.  Corpus and memory limitations prevented accurate learning by the bot, but the approach demonstrates proof of concept.
- A supervised learning approach to chatbot response discrimination was constructed based on a second seq2seq LSTM RNN.  Not surprisingly, the lack of performance by the chatbot made the discrimination task easier for the "Turing test" classifier and preformance metrics were very good.  This would not be expected to reflect real-world practice, but again provides a proof of concept of the approach.
- A second approach based on unsupervised clustering using an autoencoder was also capable of providing good accuracy in discriminating bot from human response under the project conditions.  The unsupervised approach was not as accurate as the supervised approach, but would not require as much validated entity classification information in practice. 
- Compute intensive activities performed on AWS g2.2xlarge (GPU-enabled) instance:  8 virtual core CPU Intel Xeon-E5-2670 (Sandy Bridge) with 15 GiB, 1,536 CUDA core NVIDIA Kepler GK104 GPU 4 Gb of video memory.

**Repo contents:**
- src folder includes Python 3 .py files typically applied on AWS instance and .ipynb Jupyter Notebooks used to condition and analyze data
- data folder includes source and analytic output data and saved models

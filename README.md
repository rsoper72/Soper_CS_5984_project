# Soper_CS_5984_project  

**Robert Soper, pid: rsoper**  
10 Dec 2017

## "Automating Turing’s Test: An Encoder-Decoder for Chatbot Identification"

*This work is the opinion of the author in partial fullfillment of Viriginia Tech class CS 5984 Deep Learning and does not reflect the views of the U.S. Government, the Department of Defense, or the Defense Intelligence Agency.*

**Summary rationale for this project:**  
- Chatbots can have applications for automation and efficiency in business, but increasing they are having a negative impact on the public discourse in the social media domain as evidenced by Senate hearings and academic studies of influence on the last U.S. election cycle and FCC position on net-neutrality vote "bot or not"
- Researchers at Indiana University and elsewhere have shown that traditional machine learning approaches based on a broad range of features can be effective at discriminating chatbot entities from human entities on Twitter and other social media platforms
- However, we observe that features currently most useful to traditional discrimination of chatbot from humans are the most easy to obsfucate.  Over time there may be increased effort to hide bot activity to protect their effectiveness.  More difficult will be the ability of chatbots to minimic human dialog with high accuracy, although recent advances in deep learning technique will likely continue to improve this performance.  In the future, it will likely be required to include sophisticated evaluation of entity dialog characteristics as part of bot discrimination.

**Summary of technical approach:**
- Lacking a large corpus of social media dialog data and associated metadata, this project uses the Cornell Movie Dialog corpus as a proxity to build scoped training and test datasets.

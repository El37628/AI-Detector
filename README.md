# AI Text Detection Bot


This is a Discord bot that detects AI-generated text in a given input text. It uses a machine learning model based on the Roberta architecture and various natural language processing techniques.

**Requirements**

The following packages are required to run the bot:

* re
* torch
* numpy
* os
* nltk
* contractions
* sklearn
* transformers
* nlpaug
* modAL
* scipy
* discord
* asyncio

To install the required packages, run the following command:

" pip install -r requirements.txt "

**Usage**

1. Create a Discord bot and obtain the bot token.

2. Replace [ENTER_YOUR_BOT_TOKEN] in the code with your bot token.

3. Prepare AI-generated and human-written text samples in separate text files named ai_text.txt and human_text.txt, respectively. Each sample should be on a separate line.
   
4. Run the bot using the following command:

   " python detect.py "

5. Invite the bot to your Discord server using the generated invite link.

6. In any channel, use the command !detect [text] to detect AI-generated text in the provided text.

**Functionality**

1. Detects the percentage of AI-generated and human-written text in the input.
   
2. Calculates the percentage of AI-generated text in the augmented versions of the input text using synonym augmentation.

3. Calculates the percentage of AI-generated text using TF-IDF active learning.

4. The bot responds with the calculated percentages as messages in the Discord channel.

> *Note:* The bot assumes the availability of the ai_text.txt and human_text.txt files in the same directory as the script.

For further assistance, please refer to the Discord bot documentation or contact the developer.

Discord Contact: *el37628*

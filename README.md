# OllamaSite
This is an python project that streams an python flask server. You can use an telegram bot to stream it or launch it directly  
# Main
This is an script for python. It uses flask to stream an site so that you can chat with ollama inside the site.  
The python script generates an 50 chars token for to access the main place to chat.  
You can enable ExT for extended thinking. Only works for thinking models.  
Site does have an tool function that you can enable so that the ai model can access like site or send images inside the chat that it found on sites.  
Inside data is an .txt file named benchmarks.txt that displays all the benchmarks inside the stats.  
The site does have an image generation function but because amd rocm does have an bit of faults inside i had to hard cap the generation to 824x824 but of course you can remove that or change it.  
The model generation now exist of 3 models that i have downloaded. SDXL and 2 other models.  
You can stream the site with gunicorn but if you wanna get the token you'll have to do ```DURATION={desired time} gunicorn -k gevent app:app```  
# essentials
Make sure you have ollama installed with few models to test or main models.  
Install all requirements with
```bash
pip install -r requirements
```
If you also wanna run the telegram bot dont forget to save the exports for example
```bash
export TELEGRAM_BOT_TOKEN="{given token by @BotFather}"
export TELEGRAM_USER_ID="{Userid}"
```
UserID is crutial if you don't want that any one can access the bot.  
Also when using the telegram bot make sure you have ngrok installed and ready for use with an exsiting token.  
# ALL functions
Image generation to text generation of ollama models.  
It gives an random 50 character token.  
Also default context length is 8k but inside the option menu you can lower it to 1k or to 16k max if i recall.  
Inside the site you'll get also an sandbox page for to test scripts like python or sh that an ai might give you. But be aware of what you run.  
You can also live edit scripts that it gives you and also preview other type of scripts like html.  
You can also set personalizations for the ai model like role play or if you prefer it to talk in an certain way.  
ExT helps the ai an bit for better token generations BUT it will eat up more tokens so it also might feel an bit slower.  
Also it's possible to fold out the reasoning part to see how it thinks.  
There is also an memory function so that you can force the ai to rember parts of the chats.  
There is an tts inside the browser it is currently espeak-ng on ubuntu. ```sudo apt install espeak-ng```  


Thats it. All i have to share. Find the rest your self out. You can edit it and repost it however you like. Just give an few credits.

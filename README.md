# languages

The directory consists of the following files and folders:
./instructions
In the above foloder we keep the instructions to the tutour as system prompts. 
instructions.txt file is relevant only for the first lesson at any level. There the 
level is assessed and then the schoold director teach the student
files like level_x_lesson_y.txt indicate the instructions to the tutour on how to 
proceed with the lesson based on the level and lesson, based on last 
conversaion infor stored in Postgres
For now (01.07) the next level is hardcoded to level_2_lesson_2.txt for tuning 
purposes. Once the prompts are tuned, the block in app.py should be uncommented and the 
load of the instruction for 2nd and further lessons to become dynamic

auth.py is in charge of user sign up/sign in
The info is in users table in the database

database.py is in charge to open connection and create tables in Postgres

chat.py module has two functions:
- load_last_conversation this function searches the database and loads last 
conversation, so it looks like user continues the lesson. Based on this conversation 
the tutour picks up the info about level and lesson and load the lesson plan
- update_level once during the first lesson the level of the student is identified, 
this function is called by LLM and then it uodates all stored records in postgres with 
lesson and level info. If we ever decide to identify teaching method, then this info 
will be stored there as well. Also, once the prompts are done, native language and 
language to study to be added to the database
- app.py main module, in charge of saving state, recording conversation, communicate 
with LLM 

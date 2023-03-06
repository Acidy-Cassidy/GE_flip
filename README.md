# GE_flip


OSRS Grand Exchange Trading Bot
This is a Python program that analyzes data from the Old School RuneScape (OSRS) Grand Exchange API to identify profitable trading opportunities. It filters the data based on several criteria, including trade volume, price difference, and price trend, and then presents the top 5 items to buy and their recommended sell prices.



**Requirements**
Python 3.6 or higher
pandas library
requests library
tqdm library



**Installation**
Clone the repository to your local machine using Git or download the ZIP file.
Install the required libraries by running the following command in your terminal: pip install -r requirements.txt
Run the program using the following command: python GE_flip.py



**Usage**
After running the program, you will be prompted to enter the amount of gold you want to spend (or press Enter to skip filtering).
![image](https://user-images.githubusercontent.com/30472756/223224483-2102b307-375a-4162-9a80-f6827c43b4bb.png)
(NOTE THIS WARNING IN RED IS NORMAL AND NON-CRITICAL)
The program will then retrieve the latest data from the OSRS Grand Exchange API and filter it based on the specified criteria.
The top 5 items to buy and their recommended sell prices will be displayed.
![image](https://user-images.githubusercontent.com/30472756/223224702-a48013e6-2a67-4405-9e74-7e1865bebaf8.png)


**Configuration**
The following criteria can be adjusted in the program:


**trade_volume_threshold**: The minimum trade volume required for an item to be considered.

**price_difference_threshold**: The minimum price difference between the high and low prices for an item to be considered.

**price_trend_threshold**: The percentage decrease in price required for an item to be considered a good sell opportunity.

**days_to_check**: The number of days of price history to consider.



**Contributing**
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.



PLEASE NOTE THIS WAS RAN AND TESTED ON VISUAL STUDIO 2019


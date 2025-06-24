# SQT1 - SupRep

# In-silico modeling SupRep treatment in SQT1

This model is part of the European Heart Journal (2025) submission:
"AAV9-Mediated KCNH2-Suppression-Replacement Gene Therapy in Transgenic Rabbit Models with Type 1 Short QT Syndrome"
by Saranda Nimani, ... Stefan Meier, et al. 


:file_folder: The [MMT](https://github.com/HeijmanLab/SQT1-SUPREP/tree/main/MMT) folder contains the adapted ORd model with the IKR formulations from Loewe et al. 2014.
 
:computer: :snake: The Python script to create the simulations and figures used in the paper can be found in [SupRep](https://github.com/HeijmanLab/SQT1-SUPREP/blob/main/SQT1_SupRep_script.py) and the 2D simulations can be found in [SupRep_2D_scripts](https://github.com/HeijmanLab/SQT1-SUPREP/blob/main/SupRep_SQT1_2D_norm.py).

:computer: :snake: The functions used for the above-mentioned simulations can be found in [SuoRep_functions](https://github.com/HeijmanLab/SQT1-SUPREP/blob/main/SupRep_functions.py).


## Virtual environment (Instructions for pip):

Follow the below mentioned steps to re-create te virtual environment with all the correct package versions for this project.

:exclamation: **Before creating a virtual environment please make sure you fully installed Python >3.9.5 (for Linux: -dev version) and myokit (v. 1.37) already. Please follow these steps carefully: http://myokit.org/install.** :exclamation:


***1. Clone the repo:***

https://github.com/HeijmanLab/SQT1-SUPREP.git 

***2. Create virtual environment:***

This re-creates a virtual environment with all the correct packages that were used to create the model and run the simulations. 

- Set the directory:

cd SQT1-SupRep

- Create the virtual environment:

python3 -m venv SQT1-SupRep

- Activate the environment:

On Windows: SQT1-SupRep\Scripts\activate

On macOS/Linux: source SQT1-SupRep/bin/activate

- Install packages from requirements.txt:

pip install -r requirements.txt

***3. Setup and launch spyder (or any other IDE of your liking) from your anaconda prompt:***

spyder

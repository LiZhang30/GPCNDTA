1 System requirements:

	Hardware requirements: 
		train.py requires a computer with enough RAM to support the in-memory operations.
		Operating system：windows 10/Linux

	Code dependencies:
		python '3.7' (conda install python==3.7)
		pytorch-GPU '1.10.1' (conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch)
		numpy '1.16.5' (conda install numpy==1.16.5)
		dgl (conda install -c dglteam dgl-cuda10.2)
		RDkit (conda install -c conda-forge/label/cf202003 rdkit)

2 Installation guide:

	First, install CUDA 10.2 and CUDNN 8.2.0.
	Second, install Anaconda3. Please refer to https://www.anaconda.com/distribution/ to install Anaconda3.
	Third, install PyCharm. Please refer to https://www.jetbrains.com/pycharm/download/#section=windows.
	Fourth, install dgl. Please refer to https://www.dgl.ai/.
	Fifth, install RDkit. Please refer to https://www.rdkit.org/.
	Sixth, open Anaconda Prompt to create a virtual environment by the following command:
		conda env create -n env_name -f environment.yml
	
	Note: the environment.yml file should be downloaded and put into the default path of Anaconda Prompt.

3 Instructions for use(five benchmark datasets are included in our data):

	Based on davis dataset:
		First, put folder data, cpu, utils, model and train.py into the same folder.
		Second, use PyCharm to open train.py and set the python interpreter of PyCharm.
		Third, modify codes in train.py to set parameters for the davis dataset. The details are as follows:
			line 115-116 in train.py: 'args.davis_dir, args.davis, args.smi_maxDA, args.tar_maxDA, args.train_numDA, args.test_numDA'.
		Fourth, modify codes in train.py to set emetrics for davis dataset. The details are as follows:
			line 225-228 in train.py: enable.
			line 229-232 in train.py: comment out.
		Fifth, open Anaconda Prompt and enter the following command:
			activate env_name
		Sixth, run train.py in PyCharm.

		Expected output：
			Results (mse, ci, rm2) predicted by GPCNDTA on test set of davis dataset would be output as the csv files.

	Based on kiba dataset:
		First, put folder data, cpu, utils, model and train.py into the same folder.
		Second, use PyCharm to open train.py and set the python interpreter of PyCharm.
		Third, modify codes in train.py to set parameters for the kiba dataset. The details are as follows:
			line 115-116 in train.py: 'args.kiba_dir, args.kiba, args.smi_maxKB, args.tar_maxKB, args.train_numKB, args.test_numKB'.
		Fourth, modify codes in train.py to set emetrics for kiba dataset. The details are as follows:
			line 225-228 in train.py: enable.
			line 229-232 in train.py: comment out.
		Fifth, open Anaconda Prompt and enter the following command:
			activate env_name
		Sixth, run train.py in PyCharm.

		Expected output：
			Results (mse, ci, rm2) predicted by GPCNDTA on test set of kiba dataset would be output as the csv files.

	Based on filtered davis dataset:
		First, put folder data, cpu, utils, model and train.py into the same folder.
		Second, use PyCharm to open train.py and set the python interpreter of PyCharm.
		Third, modify codes in train.py to set parameters for the filtered davis dataset. The details are as follows:
			line 115-116 in train.py: 'args.fdavis_dir, args.fdavis, args.smi_maxFDA, args.tar_maxFDA, args.train_numFDA, args.test_numFDA'.
		Fourth, modify codes in train.py to set emetrics for filtered davis dataset. The details are as follows:
			line 225-228 in train.py: comment out.
			line 229-232 in train.py: enable.
		Fifth, open Anaconda Prompt and enter the following command:
			activate env_name
		Sixth, run train.py in PyCharm.

		Expected output：
			Results (rmse, ci, spearman) predicted by GPCNDTA on test set of filtered davis dataset would be output as the csv files.

	Based on metz dataset:
		First, put folder data, cpu, utils, model and train.py into the same folder.
		Second, use PyCharm to open train.py and set the python interpreter of PyCharm.
		Third, modify codes in train.py to set parameters for the metz dataset. The details are as follows:
			line 115-116 in train.py: 'args.metz_dir, args.metz, args.smi_maxMT, args.tar_maxMT, args.train_numMT, args.test_numMT'.
		Fourth, modify codes in train.py to set emetrics for metz dataset. The details are as follows:
			line 225-228 in train.py: enable.
			line 229-232 in train.py: comment out.
		Fifth, open Anaconda Prompt and enter the following command:
			activate env_name
		Sixth, run train.py in PyCharm.

		Expected output：
			Results (mse, ci, rm2) predicted by GPCNDTA on test set of metz dataset would be output as the csv files.

	Based on toxcast dataset:
		First, put folder data, cpu, utils, model and train.py into the same folder.
		Second, use PyCharm to open train.py and set the python interpreter of PyCharm.
		Third, modify codes in train.py to set parameters for the toxcast dataset. The details are as follows:
			line 115-116 in train.py: 'args.toxcast_dir, args.toxcast, args.smi_maxTC, args.tar_maxTC, args.train_numTC, args.test_numTC'.
		Fourth, modify codes in train.py to set emetrics for toxcast dataset. The details are as follows:
			line 225-228 in train.py: enable.
			line 229-232 in train.py: comment out.
		Fifth, open Anaconda Prompt and enter the following command:
			activate env_name
		Sixth, run train.py in PyCharm.

		Expected output：
			Results (mse, ci, rm2) predicted by GPCNDTA on test set of toxcast dataset would be output as the csv files.

	Note: The five benchmark datasets are in the folder data. The folder data and model include python files related to GPCNDTA.



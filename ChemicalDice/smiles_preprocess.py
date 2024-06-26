import pandas as pd
from openbabel import pybel
#import mol2_to_image
import time
import os
from tqdm import tqdm

#from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor




# important function for mopac
def smile_to_mol2_uff(smile, steps, filename):
        mymol = pybel.readstring("smi", smile)
        #print(mymol.molwt)
        mymol.make3D(steps=steps,forcefield='uff')
        mymol.write(format="mol2",filename=filename,overwrite=True)


def smile_to_mol2_mmff(smile, steps, filename):
        mymol = pybel.readstring("smi", smile)
        #print(mymol.molwt)
        mymol.make3D(steps=steps,forcefield='mmff94')
        mymol.write(format="mol2",filename=filename,overwrite=True)


def smile_to_mol2(smiles_list,smiles_id_list, output_dir):
  mol2_file_paths = []
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  n=0
  for smiles, smiles_id in tqdm(zip(smiles_list,smiles_id_list),total=len(smiles_list)):
    n+=1
    mol2file_name = os.path.join(output_dir,str(smiles_id)+".mol2")
    uff=False
    try:
        if os.path.exists(mol2file_name):
            pass
        else:
            smile_to_mol2_mmff(smiles, steps=5000000, filename=mol2file_name)
        mol2_file_paths.append(mol2file_name)
    except:
        uff=True
    if uff==True:
        try:        
            if os.path.exists(mol2file_name):
                pass
            else:
                smile_to_mol2_uff(smiles, steps=5000000, filename=mol2file_name)
            mol2_file_paths.append(mol2file_name)
        except:
            mol2_file_paths.append("")
  return(mol2_file_paths)



def smile_to_canonical(smiles_list):
    canonical_smiles_list = []
    for smiles in smiles_list:
        try:
            molecule = pybel.readstring("smi", smiles )
            canonical_smiles = molecule.write("can").strip()
            canonical_smiles_list.append(canonical_smiles)
        except:
            canonical_smiles_list.append(canonical_smiles)
    return(canonical_smiles_list)


# def add_mol2_files(input_file,output_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     smiles_df = pd.read_csv(input_file)
#     smiles_list = smiles_df['SMILES']
#     if 'id' in smiles_df.columns:
#         smiles_id_list = smiles_df['id']
#     else:
#         smiles_df['id'] = [ "C"+str(id) for id in range(len(smiles_list))]
#         smiles_id_list = smiles_df['id']
#     mol2_file_paths = smile_to_mol2(smiles_list, smiles_id_list, output_dir)
#     smiles_df['mol2_files'] = mol2_file_paths
#     smiles_df.to_csv(input_file,index=False)
    #return(smiles_df)

def add_canonical_smiles(input_file):
    smiles_df = pd.read_csv(input_file)
    smiles_list = smiles_df['SMILES']
    if 'id' in smiles_df.columns:
        smiles_id_list = smiles_df['id']
    else:
        smiles_df['id'] = [ "C"+str(id) for id in range(len(smiles_list))]
        smiles_id_list = smiles_df['id']
    canonical_smiles_list = smile_to_canonical(smiles_list)
    smiles_df['Canonical_SMILES'] = canonical_smiles_list
    smiles_df.to_csv(input_file,index=False)

def unique(list1):
    # initialize a null list
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return(unique_list)

def line_prepender(filename, line ):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)





def create_sdf_files(input_file,output_dir="temp_data/sdffiles"):
  smiles_df = pd.read_csv(input_file)
  if not os.path.exists(output_dir):
    print("making directory ", output_dir)
    os.makedirs(output_dir)
  mol2file_name_list = smiles_df['mol2_files']
  id_list = smiles_df['id']
  sdf_list = []
  for mol2file_name,id in tqdm(zip(mol2file_name_list, id_list)):
    try:
        sdf_name = os.path.join(output_dir,id+".sdf")
        if os.path.exists(sdf_name):
            print(sdf_name," already exist")
        else:
            for mol in pybel.readfile("mol2", mol2file_name):
                mymol = mol
            mymol.write("sdf", sdf_name ,overwrite=True)
        sdf_list.append(sdf_name)
    except:
        print("Error in conversion of ", mol2file_name)
        sdf_list.append("")
  smiles_df['sdf_files'] = sdf_list
  smiles_df.to_csv(input_file,index=False)





###################   new code       ###################





# important function for mopac
def smile_to_mol2_uff(smile, steps, filename):
        mymol = pybel.readstring("smi", smile)
        #print(mymol.molwt)
        mymol.make3D(steps=steps,forcefield='uff')
        mymol.write(format="mol2",filename=filename,overwrite=True)


def smile_to_mol2_mmff(smile, steps, filename):
        mymol = pybel.readstring("smi", smile)
        #print(mymol.molwt)
        mymol.make3D(steps=steps,forcefield='mmff94')
        mymol.write(format="mol2",filename=filename,overwrite=True)

def smile_to_mol2(smiles_id_tuple):
    smiles, smiles_id, output_dir = smiles_id_tuple
    mol2_file_path = os.path.join(output_dir, str(smiles_id) + ".mol2")
    if not os.path.exists(mol2_file_path):
        try:
            smile_to_mol2_mmff(smiles, steps=500000, filename=mol2_file_path)
        except:
            try:
                smile_to_mol2_uff(smiles, steps=500000, filename=mol2_file_path)
            except:
                return ""
    return mol2_file_path

import multiprocessing

cpu_to_use = multiprocessing.cpu_count() * 0.5
cpu_to_use = int(cpu_to_use)

def create_mol2_files(input_file, output_dir="temp_data/mol2files", ncpu = cpu_to_use):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    smiles_df = pd.read_csv(input_file)
    smiles_list = smiles_df['SMILES']
    if 'id' in smiles_df.columns:
        smiles_id_list = smiles_df['id']
    else:
        smiles_df['id'] = ["C" + str(id) for id in range(len(smiles_list))]
        smiles_id_list = smiles_df['id']
    smiles_id_tuples = [(smiles, smiles_id, output_dir) for smiles, smiles_id in zip(smiles_list, smiles_id_list)]

    # with Pool() as pool:
    #     mol2_file_paths = list(tqdm(pool.imap(smile_to_mol2, smiles_id_tuples), total=len(smiles_id_tuples)))
    with ProcessPoolExecutor(max_workers=ncpu) as executor:
        mol2_file_paths = list(tqdm(executor.map(smile_to_mol2, smiles_id_tuples), total=len(smiles_id_tuples)))

    smiles_df['mol2_files'] = mol2_file_paths
    smiles_df.to_csv(input_file, index=False)


# input_file = 'pcbaexample.csv'
# output_dir = 'mol2files4'

# start_time = time.time()
# add_mol2_files(input_file, output_dir, ncpu)
# end_time = time.time()
# execution_time = end_time - start_time
# print("Script execution time: {:.2f} seconds".format(execution_time))






import os
import shutil

def make_folder_clean(folder_path):
        if os.path.exists(folder_path):
            # Remove all files and subdirectories
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
make_folder_clean(r'C:\Users\Lenovo\Desktop\final_project-main\Detected_Objects')
#make_folder_clean(r'C:\Users\Lenovo\Desktop\final_project-main\datasets\actual_data')

#make_folder_clean(r'C:\Users\Lenovo\Desktop\final_project-main\datasets\test')
#make_folder_clean(r'C:\Users\Lenovo\Desktop\final_project-main\datasets\val')
#make_folder_clean(r'C:\Users\Lenovo\Desktop\final_project-main\datasets\testing_data\f')
#make_folder_clean(r'C:\Users\Lenovo\Desktop\final_project-main\datasets\testing_data\s')
#make_folder_clean(r'C:\Users\Lenovo\Desktop\final_project-main\datap')


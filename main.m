clear

IMAGE_PATH = 'D:\umkc\courses\Neural Net\project\final\Final_Project2\'
list_dir = dir(IMAGE_PATH);

k_main = 1
main_str = ''

for i_main = 1:length(list_dir)
  names = list_dir(i_main).name
  
  if(strncmp(names, 'T', 1))
     %display('ddddddddddddddddddddddd')
     list_files = strcat(IMAGE_PATH, names)
     main_str{k_main} = names
     main_str{k_main} = main_str{k_main}(1:end-4)
     k_main = k_main + 1
     load(list_files)
  end
  
end

    
         

save('data_main.mat')
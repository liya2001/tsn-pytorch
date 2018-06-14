#!/usr/bin/env bash

# A shell script to kill PyTorch process by main process pid
# for the bug of python multiprocessing

# # Never use space in shell!!!
# # Always double quotes

# # $0 is shell file name
main_process_pid="$1"

# # awk: NR==1, only handle first line
# $12 is script name
# get python script name
py_script_name=`ps aux | grep "$main_process_pid" | awk 'NR==1 {if ($11=="python") print $12}'`

# # -n True if length of the string bigger than 0
# # -z True if length of the string equal to 0
if [ -n "$py_script_name" ];
then
    echo "Warning!!! ALL $py_script_name related process will be killed"

    # print and read input
    # # -t time; -n 1 only one character;
    read -t 3 -n 1 -p "Do you want to continue(y/n)" yn
    case $yn in
        # kill process and free GPU memory
        # # Always double semicolon
        [Yy]* ) `ps aux | grep -ie "$py_script_name" | awk '{print $2}' | xargs kill -9`;
            echo "\n$main_process_pid is killed";;
        [Nn]* ) exit;;
        * ) exit;;
    esac
else
    echo "Python script no found, please check the pid is correct!"
fi

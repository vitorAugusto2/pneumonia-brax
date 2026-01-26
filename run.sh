#!/usr/bin/env bash

echo "
██████╗ ██████╗  █████╗ ██╗  ██╗
██╔══██╗██╔══██╗██╔══██╗╚██╗██╔╝
██████╔╝██████╔╝███████║ ╚███╔╝
██╔══██╗██╔══██╗██╔══██║ ██╔██╗
██████╔╝██║  ██║██║  ██║██╔╝ ██╗
╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝

      P N E U M O N I A
"

echo "Checking dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt --upgrade-strategy only-if-needed > /dev/null

echo
echo "Running..."
python main.py

echo
echo "Completed successfully :)"
echo
read -p "Press ENTER to close..."

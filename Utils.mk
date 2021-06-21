# Source: https://techstop.github.io/bash-script-colors/
red="\e[0;91m"
blue="\e[0;94m"
expand_bg="\e[K"
blue_bg="\e[0;104m"${expand_bg}
red_bg="\e[0;101m"${expand_bg}
green_bg="\e[0;102m"${expand_bg}
green="\e[0;92m"
white="\e[0;97m"
bold="\e[1m"
uline="\e[4m"
reset="\e[0m"

define log_info
	echo -e ${blue}${bold}"[INFO]" "$(1)" ${reset}
endef

define log_error
	echo -e ${red}${bold}"[ERROR]" "$(1)" ${reset}
endef

define log_success
	echo -e ${green}${bold}"[OK]" "$(1)" ${reset}
endef
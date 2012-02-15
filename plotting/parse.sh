#!/bin/sh

cat $@ | grep "Size: " | awk -F' ' '{print $2"\t"$4}'

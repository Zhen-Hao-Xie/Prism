for i in {7..8}; do
    echo "Starting Task$i..."
    touch /data2/mnt2/zhoudw/zh/tangjt/PyMCIT/SAME_output/train/task$i.txt
    bash /data2/mnt2/zhoudw/zh/tangjt/PyMCIT/scripts/SAME/Train_CoIN/Task$i.sh > /data2/mnt2/zhoudw/zh/tangjt/PyMCIT/SAME_output/train/task$i.txt 2>&1
    echo "Task$i finished,exit with code: $?"
done
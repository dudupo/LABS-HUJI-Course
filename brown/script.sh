

for file in ./avi/*.avi; do
name=`basename "$file"`
echo "y" | ffmpeg -i $file -f  lavfi -i color=gray:s=2560x1920 -f lavfi -i color=black:s=2560x1920 -f lavfi -i color=white:s=2560x1920  -lavfi threshold -f image2 t$name%05d.png
done


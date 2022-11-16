set one=%1

if "%one%"=="" (
set remark="Update") else (
set remark=%1)


git status
git add .
git commit -m %remark%
git push origin master
import csv
import os
import shutil

#read the csv
with open('C:\\Users\\deSni\\.deeplearning4j\\data\\melanomaChallenge\\train.csv','r') as file:
    reader=csv.DictReader(file)

    #create empty list
    melanomaclass=[]
    nervusclass=[]
    seboclass=[]
    cafe_au_class=[]
    atypical_class=[]
    solar_class=[]
    unknown_class=[]
    lichenoid_class=[]
    lentigo_class=[]

#create the new list to store all the empty list
    bigclass=[melanomaclass,nervusclass,seboclass,cafe_au_class,
              atypical_class,solar_class,unknown_class,lichenoid_class,lentigo_class]

    classtype=[]

# filtering base on csv file
    for row in reader:
        if row['diagnosis'] =="melanoma":
            melanomaclass.append(row['image_name'])

        elif row['diagnosis'] =="nevus":
            nervusclass.append(row['image_name'])

        elif row['diagnosis']=="seborrheic keratosis":
            seboclass.append(row['image_name'])

        elif row['diagnosis'] == "unknown":
            unknown_class.append(row['image_name'])

        elif row['diagnosis'] =="cafe-au-lait macule":
            cafe_au_class.append(row['image_name'])

        elif row['diagnosis']=="lentigo NOS":
            lentigo_class.append(row['image_name'])

        elif row['diagnosis'] == "atypical melanocytic proliferation":
            atypical_class.append(row['image_name'])

        elif row['diagnosis'] == "lichenoid keratosis":
            lichenoid_class.append(row['image_name'])

        elif row['diagnosis'] == "solar lentigo":
            solar_class.append(row['image_name'])


#Define the unique value in diagnosis column
classtype=["melanoma","nevus","seborrheic keratosis","cafe-au-lait macule","atypical melanocytic proliferation","solar lentigo",
           "unknown","lichenoid keratosis","lentigo NOS"]

print("the class type"+str(classtype))
print("The length of big class "+str(len(bigclass)))

# modify the name from ISICxxxx to ISIC.jpg
for i in range(len(bigclass)):
    bigclass[i]=list(map(lambda x:x+'.jpg',bigclass[i]))

#Checking the class by printing
print(bigclass[0])
print(len(bigclass[0]))


#source path
sourcepath='C:\\Users\\deSni\\.deeplearning4j\\data\\melanomaChallenge\\train'

#destination path
folderSrc='C:\\Users\\deSni\\.deeplearning4j\\data\\melanomaChallenge\\trainSorted'

#create folder
for classname in classtype:
    os.mkdir(os.path.join(folderSrc,classname))


#loop though the file and compare the name
for file in os.listdir(sourcepath):
    for classnum in range(len(bigclass)):
        for picnum in range(len(bigclass[classnum])):
            print("\nMoving.......")
            if file == bigclass[classnum][picnum]:
                shutil.move(os.path.join(sourcepath,file),os.path.join(folderSrc,classtype[classnum]))

print("\nMoving File complete")









import numpy as np


# le nom de votre classe
# BayesNaif pour le modèle bayesien naif
# Knn pour le modèle des k plus proches voisins

class Knn: #nom de la class à changer

    def __init__(self, **kwargs):
    		"""
		c'est un Initializer.
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""

    def train(self, train, train_labels, distance, K): #vous pouvez rajouter d'autres attribus au besoin
        #Le k est determiner par le k-fold
		#c'est la méthode qui va entrainer votre modèle,
		#train est une matrice de type Numpy et de taille nxm, avec
		#n : le nombre d'exemple d'entrainement dans le dataset
		#m : le mobre d'attribus (le nombre de caractéristiques)

		#train_labels : est une matrice numpy de taille nx1

		#vous pouvez rajouter d'autres arguments, il suffit juste de
		#les expliquer en commentaire
        n=len(train)
        liste_unique=np.unique(train_labels)
        mod=len(liste_unique)
        confusion_matrix=np.zeros( (mod, mod) )
        for i in range(n):
            obs=train[i]
            label_obs=train_labels[i]
            res=self.predict(train, train_labels, obs, label_obs, distance, K)
            if (res==label_obs):
                confusion_matrix[res][label_obs] +=1
            else:
                confusion_matrix[res][label_obs] +=1
        if(mod==3):
            exactitude_totale=(confusion_matrix[0][0]+confusion_matrix[1][1]+confusion_matrix[2][2])/(np.sum(confusion_matrix))
            print("matrice de confusion de train\n",confusion_matrix)
            print("exactitude\n",exactitude_totale)
            for i in range(mod):
                table_confusion=np.zeros((2,2))
                a=confusion_matrix[i][i]
                b=sum(confusion_matrix[:,i])-a
                c=sum(confusion_matrix[i])-a
                d=np.sum(confusion_matrix)-(a+b+c)
                table_confusion[0][0]=a
                table_confusion[0][1]=b
                table_confusion[1][0]=c
                table_confusion[1][1]=d
                exactitude=(a+d)/(a+b+c+d)
                precision=a/(a+b)
                rappel=a/(a+c)
                print("table de confusion de la classe \n",i)
                print(table_confusion)
                print("exactitude:\n",exactitude)
                print("precision:\n",precision)
                print("rappel:\n",rappel)
        else:
            a=confusion_matrix[0][0]
            b=confusion_matrix[0][1]
            c=confusion_matrix[1][0]
            d=confusion_matrix[1][1]
            exactitude=(a+d)/(a+b+c+d)
            precision=a/(a+b)
            rappel=a/(a+c)
            print("Matrice de confusion d'entrainement:\n",confusion_matrix)		
            print("exactitude:\n",exactitude)
            print("precision:\n",precision)
            print("rappel:\n",rappel)        



		#Après avoir fait l'entrainement, faites maintenant le test sur 
		#les données d'entrainement
		#IMPORTANT : 
		#Vous devez afficher ici avec la commande print() de python,
		#- la matrice de confision (confusion matrix)
		#- l'accuracy
		#- la précision (precision)
		#- le rappel (recall)

		#Bien entendu ces tests doivent etre faits sur les données d'entrainement
		#nous allons faire d'autres tests sur les données de test dans la méthode test()

    def predict(self, train, train_labels,  exemple, label, mesure, K):
		#Prédire la classe d'un exemple donné en entrée
		#exemple est de taille 1xm
		
		#si la valeur retournée est la meme que la veleur dans label
		#alors l'exemple est bien classifié, si non c'est une missclassification
        m=len(exemple)
        n=len(train)
        distance=[]
        for i in range(n):
            d=mesure(exemple,train[i])
            distance.append(d)
			#On a pris une distance euclidienne de l'exemple a chaque observation
		#Nous avons une liste de distance entre notre exemle et toutes les observations de notre ensemble d'entrainement
		#Nous devons maintenant trier cette liste par ordre croissant
		#Cependant, ce sont les indices des observations qui nous interesse
        k=K
        dist=distance
        k_voisin=[] #liste des k voisins
        while (k!=0):
        #on cherhce le minimum
            minimum = min(dist)
            c=[i for i, j in enumerate(dist) if j == minimum]
            indice=c[0]
			#on ajoute ce minimum a notre liste de voisin
            k_voisin.append(indice)
			# On donne a ce voisin un nombre élevé
            dist[indice]=99999999
            k=k-1
        #Nous allons remplacer cette liste d'indice par le label des données d'entrainement:
        label_voisin=[]
        for i in range(len(k_voisin)):
            label_voisin.append(train_labels[k_voisin[i]])
        #Maintenant nous devons predire le label
        classe_voisin=dict((i, label_voisin.count(i)) for i in label_voisin)
        liste_voisin=sorted(classe_voisin.items(), key=lambda t: t[1])
        liste_voisin.reverse()
        prediction=liste_voisin[0][0]

        return(prediction)

    def test(self, train, train_labels, test, test_labels, distance, K):
		#On ajoute une fonction de distance en fonction du type d'attribut numerique ou binaire
		#Le k est determiner par le k-fold
        n=len(test)
        m=len(test[0])
        liste_unique=np.unique(train_labels+test_labels)
        mod=len(liste_unique)
        confusion_matrix=np.zeros( (mod, mod) )
        for i in range(n):
            exemple=test[i]
            label_exemple=test_labels[i]
            pred=self.predict(train, train_labels,  exemple, label_exemple, distance, K)
            if (pred==label_exemple):
                confusion_matrix[pred][label_exemple]+=1
            else:
                confusion_matrix[pred][label_exemple]+=1
        if(mod==3):
            exactitude=(confusion_matrix[0][0]+confusion_matrix[1][1]+confusion_matrix[2][2])/(np.sum(confusion_matrix))
            print("matrice de confusion de test\n",confusion_matrix)
            print("exactitude\n",exactitude)
            for i in range(mod):
                table_confusion=np.zeros((2,2))
                a=confusion_matrix[i][i]
                b=sum(confusion_matrix[:,i])-a
                c=sum(confusion_matrix[i])-a
                d=np.sum(confusion_matrix)-(a+b+c)
                table_confusion[0][0]=a
                table_confusion[0][1]=b
                table_confusion[1][0]=c
                table_confusion[1][1]=d
                exactitude_classe=(a+d)/(a+b+c+d)
                precision=a/(a+b)
                rappel=a/(a+c)
                print("table de confusion de la classe \n",i)
                print(table_confusion)
                print("exactitude classe:\n",exactitude_classe)
                print("precision:\n",precision)
                print("rappel:\n",rappel)
        else:
            a=confusion_matrix[0][0]
            b=confusion_matrix[0][1]
            c=confusion_matrix[1][0]
            d=confusion_matrix[1][1]
            exactitude=(a+d)/(a+b+c+d)
            precision=a/(a+b)
            rappel=a/(a+c)
            print("Matrice de confusion de test:\n",confusion_matrix)		
            print("exactitude:\n",exactitude)
            print("precision:\n",precision)
            print("rappel:\n",rappel) 

        return(exactitude)       

	#Distance euclidienne    
    def euclidienne(self,x,y):
        s=len(x)
        somme=0
        for i in range(s):
            somme=somme+(x[i]-y[i])**2
        distance=somme**(0.5)
        return(distance)

    #Mesure de dissimilarités
    def diss(self,x,y):
        sim=0
        for k in range(len(x)):
            if (x[k]==y[k]):
                sim=sim+1
        sim=sim/len(x)
        diss=1-sim
        return(diss)

    def test2(self, train, train_labels, test, test_labels, distance, K):
    	#On ajoute une fonction de distance en fonction du type d'attribut numerique ou binaire
        #Le k est determiner par le k-fold
        n=len(test)
        m=len(test[0])
        liste_unique=np.unique(train_labels+test_labels)
        mod=len(liste_unique)
        confusion_matrix=np.zeros( (mod, mod) )
        for i in range(n):
            exemple=test[i]
            label_exemple=test_labels[i]
            pred=self.predict(train, train_labels,  exemple, label_exemple, distance, K)
            if (pred==label_exemple):
                confusion_matrix[pred][label_exemple]+=1
            else:
                confusion_matrix[pred][label_exemple]+=1
        if(mod==3):
            exactitude_totale=(confusion_matrix[0][0]+confusion_matrix[1][1]+confusion_matrix[2][2])/(np.sum(confusion_matrix))
            for i in range(mod):
                table_confusion=np.zeros((2,2))
                a=confusion_matrix[i][i]
                b=sum(confusion_matrix[:,i])-a
                c=sum(confusion_matrix[i])-a
                d=np.sum(confusion_matrix)-(a+b+c)
                table_confusion[0][0]=a
                table_confusion[0][1]=b
                table_confusion[1][0]=c
                table_confusion[1][1]=d
                exactitude=(a+d)/(a+b+c+d)
                precision=a/(a+b)
                rappel=a/(a+c)
        else:
            a=confusion_matrix[0][0]
            b=confusion_matrix[0][1]
            c=confusion_matrix[1][0]
            d=confusion_matrix[1][1]
            exactitude_totale=(a+d)/(a+b+c+d)
            precision=a/(a+b)
            rappel=a/(a+c)
        return(exactitude_totale)       



    def best_k(self,train, train_labels, mesure):
        L=10
        erreur_k=[]
        liste_k=[i+1 for i in range(15)]
        for h in range(len(liste_k)):
            erreur=[]
            for i in range(L):
                #On subdivise en test et en train
                entrainement=[]
                entrainement_label=[]
                test=[]
                test_label=[]
                ratio=int(len(train)/L)
                ind1=i*ratio
                ind2=(i+1)*ratio
                index=[j for j in range(ind1,ind2)]
                for k in range(len(train)):
                    if (k in index):
                        test.append(train[k])
                        test_label.append(train_labels[k])
                    else:
                        entrainement.append(train[k])
                        entrainement_label.append(train_labels[k])
                acc=self.test2(entrainement, entrainement_label, test, test_label, mesure, liste_k[h])
                erreur.append(acc)
            moy=np.mean(erreur)
            erreur_k.append(moy)
        indice=erreur_k.index(max(erreur_k))
        meilleur_k=liste_k[indice]
        return(meilleur_k)


        

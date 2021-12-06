#ajuste de escala(INTERPOLACAO)
# importa biblioteca de processamento de imagens
import cv2

# importa biblioteca numerica
import numpy as np

def redimensiona(imagemMascara, imagemBase):

# matriz com a imagem original a ser tratada
#imagemMascara = cv2.imread(r'C:\Users\Guilherme\Documents\Projetos\images\mask_vaca_lateral.png', 1)
# matriz com a imagem que servira de modelo para as dimensoes da mascara
#imagemBase = cv2.imread(r'C:\Users\Guilherme\Documents\Projetos\images\vaquinha.jpg')

# obtem os valores da largura, altura e numero de canais da imagem base
	largura, altura, c = imagemBase.shape
# coloca os valores de largura e altura em uma variavel
	dimensoes = (altura, largura)

# interploando as dimensoes da mascara
	imagemModificada = cv2.resize(
		imagemMascara, dimensoes, interpolation = cv2.INTER_LINEAR
	#INTER_AREA se refere ao tipo de interpolacao
	)

# exibe a imagem original
#cv2.imshow("Pre-Processada", imagemMascara)
#cv2.waitKey(0)

# exibe a imagem redimensionada
#cv2.imshow("Resultado", imagemModificada)
#cv2.waitKey(0)

# salva a imagem com as dimensoes ideais para a mascara
		#cv2.imwrite('..\images', mascara.png)

	return imagemModificada
# fecha todas as janelas

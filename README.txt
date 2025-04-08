
//============================================================================================//
			Instructions for Use the Blaster Vibration Control
//============================================================================================//




//-------------------------------------------------------//
			English
//-------------------------------------------------------//

README USER:

	GIS software tool in plugin format for the Qgis program, capable of simulating vibrations, 
assisting in the safety of rock blasting, as well as plotting isolines with PPV (Peak Particle Velocity)
values ​​and calculating maximum explosive charge per hole/delay for given distances from critical structures.


  To use the application in QGIS, follow the instructions below:

	1- Download all the files in this directory;
	2- Compress them in ".ZIP" format;
	3- Open QGIS;
	4- Click on "Add-ons" in the menu bar;
	5- Click on "Manage and Install Add-ons";
	6- Click on "Install from ZIP";
	7- Search for and select the compressed files, then continue with the installation.



README DEV:

  * Copy the entire directory containing your new plugin to the QGIS plugin
    directory

  * Compile the resources file using pyrcc5

  * Run the tests (``make test``)

  * Test the plugin by enabling it in the QGIS plugin manager

  * Customize it by editing the implementation file: ``vibration_control.py`
 
  * Create your own custom icon, replacing the default icon.png`

  * Modify your user interface by opening VibrationControl_dialog_base.ui in Qt Designer

  * You can use the Makefile to compile your Ui and resource files when
    you make changes. This requires GNU make (gmake)



//-------------------------------------------------------//
			Português
//-------------------------------------------------------//

LEIA-ME (Usuário):

	Ferramenta em software GIS no formato de plugin para o programa Qgis, capaz de simular vibrações,
auxiliar na segurança do desmonte de rocha além de plotar isolinhas com valores de PPV (Velocidade Pico de Partícula)
e calcular carga máxima de explosivo por furo/retardo para dadas distâncias de estruturas críticas.


  Para utilizar a aplicação no QGIS siga as instruções, abaixo:
	
	1- Fazer o download de todos os arquivos deste diretório;
	2- Compacta-los no formato ".ZIP";
	3- Abrir o QGIS;
	4- Clicar em "Complementos" na barra de menus;
	5- Clicar em "Gerenciar e Instalar Complementos";
	6- Clicar em "Instalar a partir do ZIP";
	7- Buscar e selecionar os arquivos compactados, depois siga com a instalação.
		


Leia-me (Desenvolvedor):

	* Copie o diretório inteiro contendo seu novo plugin para o diretório de plugins
	QGIS

	* Compile o arquivo de recursos usando pyrcc5

	* Execute os testes (``make test``)

	* Teste o plugin habilitando-o no gerenciador de plugins QGIS

	* Personalize-o editando o arquivo de implementação: ``vibration_control.py`

	* Crie seu próprio ícone personalizado, substituindo o icon.png padrão`

	* Modifique sua interface de usuário abrindo VibrationControl_dialog_base.ui no Qt Designer

	* Você pode usar o Makefile para compilar seus arquivos de IU e recursos quando
	fizer alterações. Isso requer GNU make (gmake)




//============================================================================================//

For more information, see the PyQGIS Developer Cookbook at:
http://www.qgis.org/pyqgis-cookbook/index.html

(C) 2011-2018 GeoApt LLC - geoapt.com

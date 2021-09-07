/* cd <Github repo/scripts>
 * source ps7_debug_setup.tcl
 * configparams force-mem-accesses 1 */

/***************************** Include Files *********************************/

#include "xparameters.h"	/* SDK generated parameters */
#include "xsdps.h"		/* SD device driver */
#include "xil_printf.h"
#include "ff.h"
#include "xil_cache.h"
#include "xplatform_info.h"
#include "xil_exception.h"
#include "xdebug.h"
#include "xscugic.h"
#include "xspike_deepclassifier.h"
#include "xtime_l.h"

/************************** Constant Definitions *****************************/
#define SPIKE_SORT_DEVICE_ID    XPAR_SPIKE_DEEPCLASSIFIER_0_DEVICE_ID

#define INTC_DEVICE_ID          XPAR_PS7_SCUGIC_0_DEVICE_ID
#define INTR_ID			XPAR_FABRIC_SPIKE_DEEPCLASSIFIER_0_INTERRUPT_INTR
#define INTC			XScuGic
#define INTC_HANDLER	        XScuGic_InterruptHandler

/**************************** Type Definitions *******************************/

/***************** Macros (Inline Functions) Definitions *********************/

/************************** Function Prototypes ******************************/
int readSDCard(void);
int writeSDCard(void);
static void IntrHandler(void *Callback);

static int SetupIntrSystem(INTC * IntcInstancePtr,
			   u16 IntrId);
static void DisableIntrSystem(INTC * IntcInstancePtr,
					u16 IntrId);

/************************** Variable Definitions *****************************/
/*
 * Device instance definitions
 */

static XSpike_deepclassifier Spike_deepclassifier;	/* Instance of the XAxiDma */
static INTC Intc;	/* Instance of the Interrupt Controller */

/*
 * Flags interrupt handlers use to notify the application context the events.
 */
volatile int Done;

static FIL fil;		/* File object */
static FATFS fatfs;
/*
 * To test logical drive 0, FileName should be "0:/<File name>" or
 * "<file_name>". For logical drive 1, FileName should be "1:/<file_name>"
 */
//static char FileName[32] = "Test.bin";
static char FileName[32] = "spikepre.bin";
static char *SD_File;

u8 MemAddress[20*1024*1024] __attribute__ ((aligned(32)));
//u8 SourceAddress[20*1024*1024] __attribute__ ((aligned(32)));

/*****************************************************************************/
/**
*
* Main function 
*
* @param	None
*
* @return	XST_SUCCESS if successful, otherwise XST_FAILURE.
*
* @note		None
*
******************************************************************************/
int main(void)
{
	uint32_t exec_time;
	int Status;
	XSpike_deepclassifier_Config *Config;
	u32 ready;
	XTime start_time;
	XTime *p_start_time= &start_time;
	XTime end_time;
	XTime *p_end_time = &end_time;

	xil_printf("\r\n--- Entering main() --- \r\n");

	/* Read from SD Card  */
	xil_printf("Reading from SD Card \r\n");

	Status = readSDCard();
	if (Status != XST_SUCCESS) {
		xil_printf("SD read failed \r\n");
		return XST_FAILURE;
	}

	xil_printf("Successfully read SD Card \r\n");
	xil_printf("%x\r\n", MemAddress);

	/* Set up Interrupt system  */
	Status = SetupIntrSystem(&Intc, INTR_ID);
	if (Status != XST_SUCCESS) {

		xil_printf("Failed intr setup\r\n");
		return XST_FAILURE;
	}

	/* Configure Spike Deep Classifier  */
	Config = XSpike_deepclassifier_LookupConfig(SPIKE_SORT_DEVICE_ID);
	if (!Config) {
		xil_printf("No config found for %d\r\n", SPIKE_SORT_DEVICE_ID);

		return XST_FAILURE;
	}

	/* Initialize Spike Deep Classifier */
	Status = XSpike_deepclassifier_CfgInitialize(&Spike_deepclassifier, Config);

	if (Status != XST_SUCCESS) {
		xil_printf("Initialization failed %d\r\n", Status);
		return XST_FAILURE;
	}

	/* Set parameters */
    XSpike_deepclassifier_Set_deeptector_mem_params_params_offset(&Spike_deepclassifier, 0 + (int)MemAddress);
    XSpike_deepclassifier_Set_deeptector_mem_params_mem_0_offset(&Spike_deepclassifier, 9505168 + (int)MemAddress);
    XSpike_deepclassifier_Set_deeptector_mem_params_mem_1_offset(&Spike_deepclassifier, 9601168 + (int)MemAddress);
    XSpike_deepclassifier_Set_bar_mem_params_params_offset(&Spike_deepclassifier, 1798348 + (int)MemAddress);
	XSpike_deepclassifier_Set_bar_mem_params_mem_0_offset(&Spike_deepclassifier, 9505168 + (int)MemAddress);
	XSpike_deepclassifier_Set_bar_mem_params_mem_1_offset(&Spike_deepclassifier, 9601168 + (int)MemAddress);
    XSpike_deepclassifier_Set_outputs_offset(&Spike_deepclassifier, 9697168 + (int)MemAddress);

//    u32 base, high, total, width, depth, length;
    u32 length;
//    base = XSpike_deepclassifier_Get_electrodes_addr_offset_BaseAddress(&Spike_deepclassifier);
//    high = XSpike_deepclassifier_Get_electrodes_addr_offset_HighAddress(&Spike_deepclassifier);
//    total = XSpike_deepclassifier_Get_electrodes_addr_offset_TotalBytes(&Spike_deepclassifier);
//    width = XSpike_deepclassifier_Get_electrodes_addr_offset_BitWidth(&Spike_deepclassifier);
//    depth = XSpike_deepclassifier_Get_electrodes_addr_offset_Depth(&Spike_deepclassifier);

//    int electrode_addr[5] = {10627792, 10635472, 10639312, 0, 0};
    int electrode_addr[5] = {9493648 + (int)MemAddress, 9501328 + (int)MemAddress, 9505168 + (int)MemAddress, 0, 0};
    length = XSpike_deepclassifier_Write_electrodes_addr_offset_Words(&Spike_deepclassifier, 0, electrode_addr, 5);
//    xil_printf("%d, %d, %d, %d, %d, %d \r\n", base, high, total, width, depth, length);
    xil_printf("Successfully wrote %d words to electrode address offset\r\n", length);

    XSpike_deepclassifier_Set_n_electrodes(&Spike_deepclassifier, 2);

	/* Enable Interrupts */
	XSpike_deepclassifier_InterruptGlobalEnable(&Spike_deepclassifier);
	XSpike_deepclassifier_InterruptEnable(&Spike_deepclassifier, 3);

	/* Start timer */
	xil_printf("Starting timer.\r\n");
	XTime_GetTime(p_start_time);

	/* Start process */
	while(1) {
		ready = XSpike_deepclassifier_IsReady(&Spike_deepclassifier);
		if (ready) {
			xil_printf("Module ready. Starting.\r\n");
			XSpike_deepclassifier_Start(&Spike_deepclassifier);
			break;
		}
	}

	/* Initialize flags  */
	Done = 0;
	while(!Done) {
	}

	/* Stop timer */
	XTime_GetTime(p_end_time);
	exec_time = (u64) end_time - (u64) start_time;
	xil_printf("Test time = 0x%x\n\r", exec_time);

	/* Write to SD Card  */
//	xil_printf("Writing to SD Card \r\n");
//	Status = writeSDCard();
//	if (Status != XST_SUCCESS) {
//		xil_printf("SD write failed \r\n");
//		return XST_FAILURE;
//	}
//
//	xil_printf("Successfully wrote SD Card \r\n");

	/* Disable Interrupt system  */
	DisableIntrSystem(&Intc, INTR_ID);



	return XST_SUCCESS;

}

/*****************************************************************************/
/**
* Read SD Card
*
* @param	None
*
* @return	XST_SUCCESS if successful, otherwise XST_FAILURE.
*
* @note		None
*
******************************************************************************/
int readSDCard(void)
{
	FRESULT Res;
	UINT NumBytesRead;

//	u32 FileSize = 9699328;
	u32 FileSize = 9697472;

	TCHAR *Path = "0:/";


	/*
	 * Register volume work area, initialize device
	 */
	Res = f_mount(&fatfs, Path, 0);

	if (Res != FR_OK) {
		return XST_FAILURE;
	}


	/*
	 * Open file with required permissions.
	 */
	SD_File = (char *)FileName;

	Res = f_open(&fil, SD_File, FA_READ);
	if (Res) {
		xil_printf("Test failed \r\n");
		return XST_FAILURE;
	}

	/*
	 * Pointer to beginning of file .
	 */
	Res = f_lseek(&fil, 0);
	if (Res) {
		return XST_FAILURE;
	}

	/*
	 * Read data from file.
	 */
	Res = f_read(&fil, (void*)MemAddress, FileSize,
			&NumBytesRead);
	if (Res) {
		return XST_FAILURE;
	}

	/*
	 * Close file.
	 */
	Res = f_close(&fil);
	if (Res) {
		return XST_FAILURE;
	}

	return XST_SUCCESS;
}

/*****************************************************************************/
/**
* Write SD Card
*
* @param	None
*
* @return	XST_SUCCESS if successful, otherwise XST_FAILURE.
*
* @note		None
*
******************************************************************************/
int writeSDCard(void)
{
	FRESULT Res;
	UINT NumBytesWritten;
	BYTE work[FF_MAX_SS];

	u32 FileSize = 9697472;


	/*
	 * To test logical drive 0, Path should be "0:/"
	 * For logical drive 1, Path should be "1:/"
	 */
	TCHAR *Path = "0:/";


	/*
	 * Register volume work area, initialize device
	 */
	Res = f_mount(&fatfs, Path, 0);

	if (Res != FR_OK) {
		return XST_FAILURE;
	}

	/*
	 * Path - Path to logical driver, 0 - FDISK format.
	 * 0 - Cluster size is automatically determined based on Vol size.
	 */
	Res = f_mkfs(Path, FM_FAT32, 0, work, sizeof work);
	if (Res != FR_OK) {
		return XST_FAILURE;
	}

	/*
	 * Open file with required permissions.
	 * Here - Creating new file with read/write permissions. .
	 * To open file with write permissions, file system should not
	 * be in Read Only mode.
	 */
	SD_File = (char *)FileName;

	Res = f_open(&fil, SD_File, FA_CREATE_ALWAYS | FA_WRITE);
	if (Res) {
		return XST_FAILURE;
	}

	/*
	 * Pointer to beginning of file .
	 */
	Res = f_lseek(&fil, 0);
	if (Res) {
		return XST_FAILURE;
	}

	/*
	 * Write data to file.
	 */
	Res = f_write(&fil, (const void*)MemAddress, FileSize,
			&NumBytesWritten);
	if (Res) {
		return XST_FAILURE;
	}

	/*
	 * Close file.
	 */
	Res = f_close(&fil);
	if (Res) {
		return XST_FAILURE;
	}

	return XST_SUCCESS;
}

/*****************************************************************************/
/*
*
* This is the Interrupt handler function.
*
* It gets the interrupt status from the hardware, acknowledges it, and if any
* error happens, it resets the hardware. Otherwise, if a completion interrupt
* is present, then sets the TxDone.flag
*
* @param	Callback is a pointer to TX channel of the DMA engine.
*
* @return	None.
*
* @note		None.
*
******************************************************************************/
static void IntrHandler(void *Callback)
{
	Done = 1;
	xil_printf("Spike sorting is done\n");
}

/*****************************************************************************/
/*
*
* This function setups the interrupt system so interrupts can occur for the
* DMA, it assumes INTC component exists in the hardware system.
*
* @param	IntcInstancePtr is a pointer to the instance of the INTC.
* @param	IntrId is the  Interrupt ID.
*
* @return
*		- XST_SUCCESS if successful,
*		- XST_FAILURE.if not succesful
*
* @note		None.
*
******************************************************************************/
static int SetupIntrSystem(INTC * IntcInstancePtr, u16 IntrId)
{
	int Status;
	XScuGic_Config *IntcConfig;


	/*
	 * Initialize the interrupt controller driver so that it is ready to
	 * use.
	 */
	IntcConfig = XScuGic_LookupConfig(INTC_DEVICE_ID);
	if (NULL == IntcConfig) {
		xil_printf("No config found for %d\n", INTC_DEVICE_ID);
		return XST_FAILURE;
	}

	Status = XScuGic_CfgInitialize(IntcInstancePtr, IntcConfig,
					IntcConfig->CpuBaseAddress);
	if (Status != XST_SUCCESS) {
		xil_printf("Initialization failed\n");
		return XST_FAILURE;
	}


	XScuGic_SetPriorityTriggerType(IntcInstancePtr, IntrId, 0xA0, 0x3);

	/*
	 * Connect the device driver handler that will be called when an
	 * interrupt for the device occurs, the handler defined above performs
	 * the specific interrupt processing for the device.
	 */
	Status = XScuGic_Connect(IntcInstancePtr, IntrId,
				(Xil_InterruptHandler)IntrHandler,
				NULL);
	if (Status != XST_SUCCESS) {
		xil_printf("Interrupt connection failed\n");
		return Status;
	}

	XScuGic_Enable(IntcInstancePtr, IntrId);


	/* Enable interrupts from the hardware */

	Xil_ExceptionInit();
	Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_INT,
			(Xil_ExceptionHandler)INTC_HANDLER,
			(void *)IntcInstancePtr);

	Xil_ExceptionEnable();

	return XST_SUCCESS;
}

/*****************************************************************************/
/**
*
* This function disables the interrupts for DMA engine.
*
* @param	IntcInstancePtr is the pointer to the INTC component instance
* @param	IntrId is interrupt ID
*
* @return	None.
*
* @note		None.
*
******************************************************************************/
static void DisableIntrSystem(INTC * IntcInstancePtr,
					u16 IntrId)
{
	XScuGic_Disconnect(IntcInstancePtr, IntrId);
}

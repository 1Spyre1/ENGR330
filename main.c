#include <msp430.h> 
#include <stdint.h>
#include <stdbool.h>
#include "bme280.h"
#include "bme280_defs.h"

//******************************************************************************
// Define Commands ************************************************************
//******************************************************************************

struct bme280_dev dev;
struct bme280_data comp_data;

#define BME280_RESET       0xE0
#define BME280_RESET_CMD   0xB6

#define BME280_CTRL_HUM    0xF2
#define BME280_CTRL_MEAS   0xF4
#define BME280_CONFIG      0xF5

#define BME280_PRESS_MSB   0xF7 



#define SLAVE_ADDR  0x76

#define MAX_BUFFER_SIZE     20

//******************************************************************************
// General I2C State Machine ***************************************************
//******************************************************************************

typedef enum I2C_ModeEnum{
    IDLE_MODE,
    NACK_MODE,
    TX_REG_ADDRESS_MODE,
    RX_REG_ADDRESS_MODE,
    TX_DATA_MODE,
    RX_DATA_MODE,
    SWITCH_TO_RX_MODE,
    SWITHC_TO_TX_MODE,
    TIMEOUT_MODE
} I2C_Mode;


/* Used to track the state of the software state machine*/
I2C_Mode MasterMode = IDLE_MODE;

/* The Register Address/Command to use*/
uint8_t TransmitRegAddr = 0;

/* ReceiveBuffer: Buffer used to receive data in the ISR
 * RXByteCtr: Number of bytes left to receive
 * ReceiveIndex: The index of the next byte to be received in ReceiveBuffer
 * TransmitBuffer: Buffer used to transmit data in the ISR
 * TXByteCtr: Number of bytes left to transfer
 * TransmitIndex: The index of the next byte to be transmitted in TransmitBuffer
 * */
uint8_t ReceiveBuffer[MAX_BUFFER_SIZE] = {0};
uint8_t RXByteCtr = 0;
uint8_t ReceiveIndex = 0;
uint8_t TransmitBuffer[MAX_BUFFER_SIZE] = {0};
uint8_t TXByteCtr = 0;
uint8_t TransmitIndex = 0;


I2C_Mode I2C_Master_WriteReg(uint8_t dev_addr, uint8_t reg_addr, uint8_t *reg_data, uint8_t count);

I2C_Mode I2C_Master_ReadReg(uint8_t dev_addr, uint8_t reg_addr, uint8_t count);
void CopyArray(uint8_t *source, uint8_t *dest, uint8_t count);


I2C_Mode I2C_Master_ReadReg(uint8_t dev_addr, uint8_t reg_addr, uint8_t count)
{
    /* Initialize state machine */
    MasterMode = TX_REG_ADDRESS_MODE;
    TransmitRegAddr = reg_addr;
    RXByteCtr = count;
    TXByteCtr = 0;
    ReceiveIndex = 0;
    TransmitIndex = 0;

    /* Initialize slave address and interrupts */
    UCB0I2CSA = dev_addr;
    UCB0IFG &= ~(UCTXIFG + UCRXIFG);       // Clear any pending interrupts
    UCB0IE &= ~UCRXIE;                       // Disable RX interrupt
    UCB0IE |= UCTXIE;                        // Enable TX interrupt

    UCB0CTLW0 |= UCTR + UCTXSTT;             // I2C TX, start condition
    __bis_SR_register(LPM0_bits + GIE);              // Enter LPM0 w/ interrupts

    return MasterMode;

}


I2C_Mode I2C_Master_WriteReg(uint8_t dev_addr, uint8_t reg_addr, uint8_t *reg_data, uint8_t count)
{
    /* Initialize state machine */
    MasterMode = TX_REG_ADDRESS_MODE;
    TransmitRegAddr = reg_addr;

    //Copy register data to TransmitBuffer
    CopyArray(reg_data, TransmitBuffer, count);

    TXByteCtr = count;
    RXByteCtr = 0;
    ReceiveIndex = 0;
    TransmitIndex = 0;

    /* Initialize slave address and interrupts */
    UCB0I2CSA = dev_addr;
    UCB0IFG &= ~(UCTXIFG + UCRXIFG);       // Clear any pending interrupts
    UCB0IE &= ~UCRXIE;                       // Disable RX interrupt
    UCB0IE |= UCTXIE;                        // Enable TX interrupt

    UCB0CTLW0 |= UCTR + UCTXSTT;             // I2C TX, start condition
    __bis_SR_register(LPM0_bits + GIE);              // Enter LPM0 w/ interrupts

    return MasterMode;
}

void CopyArray(uint8_t *source, uint8_t *dest, uint8_t count)
{
    uint8_t copyIndex = 0;
    for (copyIndex = 0; copyIndex < count; copyIndex++)
    {
        dest[copyIndex] = source[copyIndex];
    }
}

//******************************************************************************
// UART-HC05 Initialization *******************************************************
//******************************************************************************

void UART_init(void)
{
    // P2.5 = UCA1TXD, P2.6 = UCA1RXD
    P2SEL1 |= BIT5 | BIT6;
    P2SEL0 &= ~(BIT5 | BIT6);

    UCA1CTLW0 = UCSWRST;
    UCA1CTLW0 |= UCSSEL__SMCLK;
    UCA1BRW = 104; 
    UCA1MCTLW = 0xD600;
    UCA1CTLW0 &= ~UCSWRST;
    UCA1IE |= UCRXIE; 
}


void UART_print(const char *msg)
{
    while (*msg)
    {
        while (!(UCA1IFG & UCTXIFG));
        UCA1TXBUF = *msg++;
    }
}

void UART_print_hex8(uint8_t v)
{
    const char hex[] = "0123456789ABCDEF";
    while (!(UCA1IFG & UCTXIFG));
    UCA1TXBUF = hex[v >> 4];
    while (!(UCA1IFG & UCTXIFG));
    UCA1TXBUF = hex[v & 0x0F];
}

void UART_print_int(int32_t n) {
    char buf[12];
    int i = 0;
    if (n == 0) {
        UART_print("0");
        return;
    }
    if (n < 0) {
        UART_print("-");
        n = -n;
    }
    while (n > 0) {
        buf[i++] = (n % 10) + '0';
        n /= 10;
    }
    while (i > 0) {
        while (!(UCA1IFG & UCTXIFG));
        UCA1TXBUF = buf[--i];
    }
}

//******************************************************************************
// I2C Helpers *******************************************************
//******************************************************************************


int8_t i2c_read_wrapper(uint8_t reg_addr, uint8_t *reg_data, uint32_t len, void *intf_ptr) {
    // 1. Trigger the TI I2C Read
    I2C_Master_ReadReg(SLAVE_ADDR, reg_addr, (uint8_t)len);
    
    // 2. The TI code enters LPM0 and waits for the ISR to finish. 
    // Once it returns here, the data is in ReceiveBuffer.
    CopyArray(ReceiveBuffer, reg_data, (uint8_t)len);
    return BME280_OK;
}

int8_t i2c_write_wrapper(uint8_t reg_addr, const uint8_t *reg_data, uint32_t len, void *intf_ptr) {
    // Trigger the TI I2C Write
    I2C_Master_WriteReg(SLAVE_ADDR, reg_addr, (uint8_t *)reg_data, (uint8_t)len);
    return BME280_OK;
}

void delay_us_wrapper(uint32_t period, void *intf_ptr)
{
    while (period--)
    {
        __delay_cycles(16);
    }
}


void sleep_ms(uint16_t ms)
{
    if (ms > 6000) ms = 6000;

    TA0CTL |= TACLR;
    TA0CCR0 = ms * 10;

    TA0CCTL0 &= ~CCIFG;
    TA0CTL |= MC__UP;

    __bis_SR_register(LPM3_bits | GIE);

    TA0CTL &= ~MC__UP;
}


//******************************************************************************
// Device Initialization *******************************************************
//******************************************************************************


void initGPIO()
{
    // PA = P1 + P2
    PADIR = 0xFFFF; PAOUT = 0x0000;
    // PB = P3 + P4
    PBDIR = 0xFFFF; PBOUT = 0x0000;
    // PJ (JTAG/XT1)
    PJDIR = 0xFFFF; PJOUT = 0x0000;
    // Configure GPIO
    
    P1OUT &= ~BIT0;                           // Clear P1.0 output latch
    P1DIR |= BIT0;                            // For LED
    
    P1SEL0 &= ~(BIT6 | BIT7);
    P1SEL1 |= BIT6 | BIT7;                    // I2C pins
    // Disable the GPIO power-on default high-impedance mode to activate
    // previously configured port settings
    PM5CTL0 &= ~LOCKLPM5;
}

void initClockTo16MHz()
{
    // Configure one FRAM waitstate as required by the device datasheet for MCLK
    // operation beyond 8MHz _before_ configuring the clock system.
    FRCTL0 = FRCTLPW | NWAITS_1;

    // Clock System Setup
    CSCTL0_H = CSKEY >> 8;                    // Unlock CS registers
    CSCTL1 = DCORSEL | DCOFSEL_4;             // Set DCO to 16MHz
    CSCTL2 = SELA__VLOCLK | SELS__DCOCLK | SELM__DCOCLK;
    CSCTL3 = DIVA__1 | DIVS__1 | DIVM__1;     // Set all dividers

    CSCTL0_H = 0;                             // Lock CS registerss
}

void initClockTo8MHz()
{
    // Configure FRAM wait state to 0 (Valid for up to 8MHz)
    FRCTL0 = FRCTLPW | NWAITS_0; 

    CSCTL0_H = CSKEY >> 8;                    // Unlock CS registers
    CSCTL1 = DCOFSEL_3 | DCORSEL;             // Set DCO to 8MHz
    CSCTL2 = SELA__VLOCLK | SELS__DCOCLK | SELM__DCOCLK;
    CSCTL3 = DIVA__1 | DIVS__1 | DIVM__1;     // Dividers to 1
    CSCTL0_H = 0;                             // Lock CS registers
}

void initClockTo1MHz()
{
    // 1MHz is slow enough for 0 wait states (Max efficiency)
    FRCTL0 = FRCTLPW | NWAITS_0;

    CSCTL0_H = CSKEY >> 8;                    // Unlock CS registers
    CSCTL1 = DCOFSEL_0;                       // 1MHz (DCOFSEL_0 is typically 1MHz)
                                              // Note: DCORSEL is 0 by default for low freq
    CSCTL2 = SELA__VLOCLK | SELS__DCOCLK | SELM__DCOCLK;
    CSCTL3 = DIVA__1 | DIVS__1 | DIVM__1;     // No dividers
    CSCTL0_H = 0;                             // Lock CS registers
}

void initI2C()
{
    UCB0CTLW0 = UCSWRST;                      // Enable SW reset
    UCB0CTLW0 |= UCMODE_3 | UCMST | UCSSEL__SMCLK | UCSYNC; // I2C master mode, SMCLK
    UCB0BRW = 10;                            
    UCB0I2CSA = SLAVE_ADDR;                   // Slave Address
    UCB0CTLW0 &= ~UCSWRST;                    // Clear SW reset, resume operation
    UCB0IE |= UCNACKIE;
}

void initTimerA(void)
{
    TA0CTL = TASSEL__ACLK | MC__STOP | TACLR;
    TA0CCTL0 = CCIE;
    TA0CCTL0 &= ~CCIFG;
}

int8_t setup_bme280() {
    int8_t rslt1;
    int8_t rslt2;


    dev.intf = BME280_I2C_INTF;
    dev.read = i2c_read_wrapper;
    dev.write = i2c_write_wrapper;
    dev.delay_us = delay_us_wrapper;
    dev.intf_ptr = NULL; // Not used for this hardware

    rslt1 = bme280_init(&dev);

    // Set configuration: Normal Mode, 1x Oversampling
    dev.settings.osr_h = BME280_OVERSAMPLING_1X;
    dev.settings.osr_p = BME280_OVERSAMPLING_1X;
    dev.settings.osr_t = BME280_OVERSAMPLING_1X;
    dev.settings.filter = BME280_FILTER_COEFF_OFF;
    dev.settings.standby_time = BME280_STANDBY_TIME_62_5_MS;

    rslt2 = bme280_set_sensor_settings(BME280_ALL_SETTINGS_SEL, &dev);
    //rslt2 = bme280_set_sensor_mode(BME280_NORMAL_MODE, &dev);

    if (rslt1 != BME280_OK) return rslt1;
    if (rslt2 != BME280_OK) return rslt2;
    return BME280_OK;
}


//******************************************************************************
// Main ************************************************************************
// Send and receive three messages containing the example commands *************
//******************************************************************************

int main(void) {
    WDTCTL = WDTPW | WDTHOLD;	// Stop watchdog timer
    initClockTo1MHz();
    initGPIO();
    initI2C();
    UART_init();
    initTimerA();
    int8_t rslt;
    UART_print("Initializations were completed...");

    rslt = setup_bme280();
    if (rslt == BME280_OK) {
        UART_print("BME280 OK\r\n");
    } else {
        UART_print("BME280 INIT ERROR\r\n");
        UART_print_int((int32_t)rslt);
    }

    uint8_t chip_id = 0;
    i2c_read_wrapper(0xD0, &chip_id, 1, &dev);

    UART_print("Chip ID: 0x");
    UART_print_hex8(chip_id);
    UART_print("\r\n");

    while(1) {


            bme280_set_sensor_mode(BME280_FORCED_MODE, &dev);
            
            sleep_ms(10);
            
            // Read all sensor data
            
            int8_t rslt3;
            rslt3 = bme280_get_sensor_data(BME280_ALL, &comp_data, &dev);
    
            if (rslt3 != BME280_OK) {
                UART_print("READ ERR\r\n");
            }

            // Send to HC-05
            UART_print("Temp: ");
            // Note: Bosch provides data in hundredths (e.g., 2550 = 25.50 C)
            // You might need a simple itoa or print function for the HC-05
            UART_print_int((int32_t)(comp_data.temperature)); UART_print("."); UART_print_int((int32_t)(comp_data.temperature*100)%100); 
            UART_print(" C | Pres: ");
            UART_print_int((int32_t)(comp_data.pressure/100)); UART_print("."); UART_print_int((int32_t)(comp_data.pressure)%100); 
            UART_print(" hPa | Hum: ");
            UART_print_int((int32_t)(comp_data.humidity)); UART_print("."); UART_print_int((int32_t)(comp_data.humidity*100)%100); 
            UART_print(" %");
            // Pressure is in Pascals
            
            UART_print("\r\n");

            sleep_ms(1000);
            // delay_us_wrapper(1000000, NULL); // Wait 1 second
        }

}


//******************************************************************************
// I2C Interrupt ***************************************************************
//******************************************************************************


#pragma vector = TIMER0_A0_VECTOR
__interrupt void TIMER0_A0_ISR(void)
{
    __bic_SR_register_on_exit(LPM3_bits);
}

#if defined(__TI_COMPILER_VERSION__) || defined(__IAR_SYSTEMS_ICC__)
#pragma vector = USCI_B0_VECTOR
__interrupt void USCI_B0_ISR(void)
#elif defined(__GNUC__)
void __attribute__ ((interrupt(USCI_B0_VECTOR))) USCI_B0_ISR (void)
#else
#error Compiler not supported!
#endif
{
  //Must read from UCB0RXBUF
  uint8_t rx_val = 0;
  switch(__even_in_range(UCB0IV, USCI_I2C_UCBIT9IFG))
  {
    case USCI_NONE:          break;         // Vector 0: No interrupts
    case USCI_I2C_UCALIFG:   break;         // Vector 2: ALIFG
    case USCI_I2C_UCNACKIFG:                // Vector 4: NACKIFG
      break;
    case USCI_I2C_UCSTTIFG:  break;         // Vector 6: STTIFG
    case USCI_I2C_UCSTPIFG:  break;         // Vector 8: STPIFG
    case USCI_I2C_UCRXIFG3:  break;         // Vector 10: RXIFG3
    case USCI_I2C_UCTXIFG3:  break;         // Vector 12: TXIFG3
    case USCI_I2C_UCRXIFG2:  break;         // Vector 14: RXIFG2
    case USCI_I2C_UCTXIFG2:  break;         // Vector 16: TXIFG2
    case USCI_I2C_UCRXIFG1:  break;         // Vector 18: RXIFG1
    case USCI_I2C_UCTXIFG1:  break;         // Vector 20: TXIFG1
    case USCI_I2C_UCRXIFG0:                 // Vector 22: RXIFG0
        rx_val = UCB0RXBUF;
        if (RXByteCtr)
        {
          ReceiveBuffer[ReceiveIndex++] = rx_val;
          RXByteCtr--;
        }

        if (RXByteCtr == 1)
        {
          UCB0CTLW0 |= UCTXSTP;
        }
        else if (RXByteCtr == 0)
        {
          UCB0IE &= ~UCRXIE;
          MasterMode = IDLE_MODE;
          __bic_SR_register_on_exit(CPUOFF);      // Exit LPM0
        }
        break;
    case USCI_I2C_UCTXIFG0:                 // Vector 24: TXIFG0
        switch (MasterMode)
        {
          case TX_REG_ADDRESS_MODE:
              UCB0TXBUF = TransmitRegAddr;
              if (RXByteCtr)
                  MasterMode = SWITCH_TO_RX_MODE;   // Need to start receiving now
              else
                  MasterMode = TX_DATA_MODE;        // Continue to transmision with the data in Transmit Buffer
              break;

          case SWITCH_TO_RX_MODE:
              UCB0IE |= UCRXIE;              // Enable RX interrupt
              UCB0IE &= ~UCTXIE;             // Disable TX interrupt
              UCB0CTLW0 &= ~UCTR;            // Switch to receiver
              MasterMode = RX_DATA_MODE;    // State state is to receive data
              UCB0CTLW0 |= UCTXSTT;          // Send repeated start
              if (RXByteCtr == 1)
              {
                  //Must send stop since this is the N-1 byte
                  while((UCB0CTLW0 & UCTXSTT));
                  UCB0CTLW0 |= UCTXSTP;      // Send stop condition
              }
              break;

          case TX_DATA_MODE:
              if (TXByteCtr)
              {
                  UCB0TXBUF = TransmitBuffer[TransmitIndex++];
                  TXByteCtr--;
              }
              else
              {
                  //Done with transmission
                  UCB0CTLW0 |= UCTXSTP;     // Send stop condition
                  MasterMode = IDLE_MODE;
                  UCB0IE &= ~UCTXIE;                       // disable TX interrupt
                  __bic_SR_register_on_exit(CPUOFF);      // Exit LPM0
              }
              break;

          default:
              __no_operation();
              break;
        }
        break;
    default: break;
  }
}

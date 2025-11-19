union byte_to_float {
     byte b[4];
     float f;
 };


 void stream_data(){

        byte checksum = 0xFE; // Initial checksum value

        Serial.write(0xFF); // Start byte
        Serial.write(0xFF); // Start byte
        Serial.write(9); // Payload Size (2 floats = 8 bytes) + 1 (checksum)
        
        byte_to_float data_byte;
        data_byte.f = MEnc1;
        Serial.write(data_byte.b, sizeof(data_byte.b));
        checksum += data_byte.b[0];
        checksum += data_byte.b[1];
        checksum += data_byte.b[2];
        checksum += data_byte.b[3];
        
        data_byte.f = MEnc2;
        Serial.write(data_byte.b, sizeof(data_byte.b));
        checksum += data_byte.b[0];
        checksum += data_byte.b[1];
        checksum += data_byte.b[2];
        checksum += data_byte.b[3];
        Serial.write(checksum); // Checksum byte

    }
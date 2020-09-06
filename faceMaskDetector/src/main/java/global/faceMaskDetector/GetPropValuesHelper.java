package global.faceMaskDetector;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Properties;

public class GetPropValuesHelper {
    public static String getPropValues(String property)throws IOException {
        InputStream inputStream = null;
        String value = null;

        try{
            Properties prop = new Properties();
            String propFileName = "config.properties";

            inputStream = global.faceMaskDetector.GetPropValuesHelper.class.getClassLoader().getResourceAsStream(propFileName);

            if (inputStream != null) {
                prop.load(inputStream);
            } else {
                throw new FileNotFoundException("property file '" + propFileName + "' not found in the classpath");
            }

            value = prop.getProperty(property);
        } catch (Exception  e) {
            e.printStackTrace();
        } finally {
            inputStream.close();
        }

        return value;
    }

    public static String getCheckSum(String filePath) {
        String hashValue = "";
        try (InputStream is = Files.newInputStream(Paths.get(filePath))) {
            hashValue = org.apache.commons.codec.digest.DigestUtils.md5Hex(is);
        } catch (Exception e){
            System.out.println(e.getMessage());
        }
        return hashValue;
    }
}

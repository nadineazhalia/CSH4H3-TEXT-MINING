/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tokenisasi;

/**
 *
 * @author Nadine Azhalia
 */
public class Tokenisasi {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        String indo = "Adit sedang makan ayam goreng";
        String token_indo[] = indo.split(" ");
        for (int i=0; i<token_indo.length;i++){
            System.out.print(token_indo[i]+", ");
        }
        System.out.println();
        String ing = "Adit is eating fried chicken";
        String token_ing[] = ing.split(" ");
        for (int i=0; i<token_ing.length;i++){
            System.out.print(token_ing[i]+", ");
        }
        System.out.println();
        String jpg = "私は 授業 中 に 勉強 して います";
        String token_jpg[] = jpg.split(" ");
        for (int i=0; i<token_jpg.length;i++){
            System.out.print(token_jpg[i]+", ");
        }
        System.out.println();
        String arb = "درست في الصف";
        String token_arb[] = arb.split(" ");
        for (int i=0; i<token_arb.length;i++){
            System.out.print(token_arb[i]+", ");
        }
        
    }
    
}

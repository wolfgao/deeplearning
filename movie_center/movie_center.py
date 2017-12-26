import media
import fresh_tomatoes

jurassic_world = media.Movie("Jurassic World 4", "",
	"http://flashwallpapers.com/wp-content/uploads/2015/04/Jurassic-World-Poster-2-1024x474.jpg",
	"https://www.youtube.com/watch?v=lYebyzthzIE")

coco = media.Movie("Coco Vive Tu Momento", "",
	"https://vignette.wikia.nocookie.net/disney/images/6/62/Coco_Vive_Tu_Momento_Poster.jpg/revision/latest?cb=20171016153046",
	"https://www.youtube.com/watch?v=cDTrPvhifIg&index=13&list=PLzjFbaFzsmMTE39Up7aJ4Inj75fJ2syIB")


star_world = media.Movie("Star Wars:The Last Jedi TV Trailer \"Heroes\"", "The Last Jedi",
	"https://i.ytimg.com/vi/gBjD2UEKj8E/maxresdefault.jpg",
	"https://www.youtube.com/watch?v=TYRy5bCsWF8")


x_men = media.Movie("X-Men: The New Mutants",
	"X-Men: The New Mutants Trailer 2017 - Official 2018 Movie Teaser Trailer in HD - starring Anya Taylor-Joy, Maisie Williams, Charlie Heaton - directed by Josh Boone \
	- Five young mutants, just discovering their abilities while held in a secret facility against their will, fight to escape their past sins and save themselves",
	"https://i.pinimg.com/736x/52/62/c3/5262c3bc26be1352567b0cbd01287745--storm-xmen-wolfsbane.jpg",
	"https://www.youtube.com/watch?v=Ez4-ZY9yY_U")


harrypotter = media.Movie("Harry Potter: And The Cursed Child -2018",
	"Harry Potter and the Cursed Child is a two-part West End stage play written by Jack Thorne and based on an original new story by Thorne, J.K. Rowling, \
	and John Tiffany.",
	"http://harrypotterfanzone.com/wp-content/2015/06/cursed-child-logo.jpg",
	"https://www.youtube.com/watch?v=VPIOygtubEQ")


pitch_perfect3 = media.Movie("PITCH PERFECT 3 \"Eyes On Me\"", "Comedy, Kids, Family and Animated Film, Blockbuster,  Action Movie, Blockbuster, Scifi, \
	Fantasy film and Drama...   We keep you in the know! ",
	"https://gentlemanrebellion.files.wordpress.com/2015/06/pitch_perfect2_key_art.jpg",
	"https://www.youtube.com/watch?v=LTV1Eng40NM")

movies = [jurassic_world, coco, star_world, x_men, harrypotter, pitch_perfect3]

#fresh_tomatoes.open_movies_page(movies)

print media.Movie.__doc__

print media.Movie.__name__

print fresh_tomatoes.os.__all__

print fresh_tomatoes.os.__doc__

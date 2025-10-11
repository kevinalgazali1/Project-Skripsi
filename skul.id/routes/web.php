<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\UserController;
use App\Http\Controllers\AdminController;
use App\Http\Controllers\LokerController;
use App\Http\Controllers\MitraController;
use App\Http\Controllers\DaftarController;
use App\Http\Controllers\ArtikelController;
use App\Http\Controllers\ProfileController;
use App\Http\Controllers\PelatihanController;
use App\Http\Controllers\AlumniSiswaController;
use App\Http\Controllers\SertifikasiController;
use App\Http\Controllers\Auth\RegisteredUserController;

/**
 * ✅ Route publik (tanpa login)
 */

Route::middleware('prevent-back-history', 'guest.redirect')->group(function () {
    Route::get('/login/alumni', function () {
        return view('login.loginAlumni');
    })->name('login.alumni');

    Route::get('/login/mitra', function () {
        return view('login.loginMitra');
    })->name('login.mitra');

    Route::get('/login/admin', function () {
        return view('login.loginAdmin');
    })->name('login.admin');

    Route::get('/', function () {
        return view('welcome');
    });

    // Halaman register
    // Route::get('/register/mitra', [RegisteredUserController::class, 'showMitraForm'])->name('register.mitra');
    Route::get('/register/alumni', [RegisteredUserController::class, 'showAlumniForm'])->name('register.alumni');
    Route::post('/registrasi', [RegisteredUserController::class, 'register'])->name('register.store');
});

/**
 * ✅ Route setelah login, role MITRA
 */
Route::middleware(['auth', 'role:mitra'])->group(function () {
    Route::get('/mitra/addProfile', [MitraController::class, 'addProfile'])->name('mitra.addProfile');
    Route::post('/mitra/storeProfile', [MitraController::class, 'storeProfile'])->name('mitra.storeProfile');
    Route::middleware('profile')->group(function () {
        Route::get('/mitra', [MitraController::class, 'index'])->name('mitra.index');
        Route::get('/mitra/sertifikasi', [SertifikasiController::class, 'index'])->name('mitra.sertifikasi');
        Route::post('/mitra/store-sertifikasi', [SertifikasiController::class, 'store'])->name('mitra.sertifikasi.store');
        Route::put('/mitra/update-sertifikasi/{id}', [SertifikasiController::class, 'update'])->name('mitra.sertifikasi.update');
        Route::delete('/mitra/delete-sertifikasi/{id}', [SertifikasiController::class, 'destroy'])->name('mitra.sertifikasi.destroy');
        Route::get('/mitra/sertifikasi/{id}/peserta', [SertifikasiController::class, 'getPeserta']);
        Route::get('/mitra/sertifikasi/{id}/peserta/export', [SertifikasiController::class, 'exportPesertaExcel']);
        Route::get('/mitra/profile', [MitraController::class, 'profile'])->name('mitra.profile');
        Route::get('/mitra/loker', [LokerController::class, 'index'])->name('mitra.loker');
        Route::post('/mitra/store-loker', [LokerController::class, 'store'])->name('mitra.loker.store');
        Route::put('/mitra/update-loker/{id}', [LokerController::class, 'update'])->name('mitra.loker.update');
        Route::delete('/mitra/delete-loker/{id}', [LokerController::class, 'destroy'])->name('mitra.loker.destroy');
        Route::get('/mitra/loker/{id}/peserta/export', [LokerController::class, 'exportPesertaExcel']);
        Route::get('/loker/{id}/peserta', [LokerController::class, 'getPeserta']);
        Route::get('/mitra/pelatihan', [PelatihanController::class, 'index'])->name('mitra.pelatihan');
        Route::post('/mitra/store-pelatihan', [PelatihanController::class, 'store'])->name('mitra.pelatihan.store');
        Route::put('/mitra/update-pelatihan/{id}', [PelatihanController::class, 'update'])->name('mitra.pelatihan.update');
        Route::delete('/mitra/delete-pelatihan/{id}', [PelatihanController::class, 'destroy'])->name('mitra.pelatihan.destroy');
        Route::get('/mitra/pelatihan/{id}/peserta/export', [PelatihanController::class, 'exportPesertaExcel']);
        Route::get('/mitra/profile', [MitraController::class, 'profile'])->name('mitra.profile');
        Route::put('/mitra/update-profile', [MitraController::class, 'updateProfile'])->name('mitra.updateProfile');
        Route::put('/mitra/account/update', [UserController::class, 'updateAccount'])->name('mitra.account.update');
        Route::get('/mitra/sertifikasi/export', [SertifikasiController::class, 'exportSertifikasi'])->name('mitra.sertifikasi.export');
        Route::get('/mitra/pelatihan/export', [PelatihanController::class, 'exportPelatihan'])->name('mitra.pelatihan.export');
        Route::get('/mitra/loker/export', [LokerController::class, 'exportLoker'])->name('mitra.loker.export');
    });
});

/**
 * ✅ Route setelah login, role ALUMNI
 */
Route::middleware(['auth', 'role:alumnisiswa'])->group(function () {
    Route::get('/alumni-siswa/addProfile', [AlumniSiswaController::class, 'addProfile'])->name('alumni-siswa.addProfile');
    Route::post('/alumni-siswa/storeProfile', [AlumniSiswaController::class, 'storeProfile'])->name('alumni-siswa.storeProfile');
    Route::get('alumni-siswa/get-sekolah', [AlumniSiswaController::class, 'getSekolah']);
    Route::get('alumni-siswa/cari-sekolah', [AlumniSiswaController::class, 'cariSekolah']);
    Route::middleware('profile')->group(function () {
        Route::get('/alumni-siswa', [AlumniSiswaController::class, 'index'])->name('alumni-siswa.index');
        Route::get('/alumni-siswa/profile', [AlumniSiswaController::class, 'profile'])->name('alumni-siswa.profile');
        Route::get('/alumni-siswa/sertifikasi', [AlumniSiswaController::class, 'sertifikasi'])->name('alumni-siswa.sertifikasi');
        Route::post('/alumni-siswa/store-sertifikasi/{sertifikasi}', [DaftarController::class, 'storeDaftarSertifikasi'])->name('alumni-siswa.sertifikasi.store');
        Route::post('/alumni-siswa/store-loker/{loker}', [DaftarController::class, 'storeDaftarLoker'])->name('alumni-siswa.loker.store');
        Route::post('alumni-siswa/store-pelatihan/{pelatihan}', [DaftarController::class, 'storeDaftarPelatihan'])->name('alumni-siswa.pelatihan.store');
        Route::get('/alumni-siswa/loker', [AlumniSiswaController::class, 'loker'])->name('alumni-siswa.loker');
        Route::get('/alumni-siswa/loker/search', [AlumniSiswaController::class, 'lokerSearch'])->name('alumni-siswa.loker.search');
        Route::get('/alumni-siswa/pelatihan', [AlumniSiswaController::class, 'pelatihan'])->name('alumni-siswa.pelatihan');
        Route::get('/alumni-siswa/ikatan', [AlumniSiswaController::class, 'ikatan'])->name('alumni-siswa.ikatan');
        // Route::get('/alumni-siswa/kuliah', [AlumniSiswaController::class, 'kuliah'])->name('alumni-siswa.kuliah');
        Route::get('/alumni-siswa/artikel', [AlumniSiswaController::class, 'artikel'])->name('alumni-siswa.artikel');
        Route::get('/alumni-siswa/editProfile', [AlumniSiswaController::class, 'editProfile'])->name('alumni-siswa.editProfile');
        Route::put('/alumni-siswa/update-profile', [AlumniSiswaController::class, 'updateProfile'])->name('alumni-siswa.updateProfile');
        Route::put('/alumni-siswa/account/update', [UserController::class, 'updateAccount'])->name('alumni-siswa.account.update');
    });
});

/**
 * ✅ Route setelah login, role ADMIN
 */
Route::middleware(['auth', 'role:admin'])->group(function () {
    Route::get('/admin/dashboard', [AdminController::class, 'index'])->name('admin.dashboard');
    Route::get('/admin/artikel', [AdminController::class, 'artikel'])->name('admin.artikel');
    Route::post('/admin/artikel/store', [ArtikelController::class, 'store'])->name('admin.artikel.store');
    Route::put('/admin/artikel/update/{id}', [ArtikelController::class, 'update'])->name('admin.artikel.update');
    Route::delete('/admin/delete-artikel/{id}', [ArtikelController::class, 'destroy'])->name('admin.artikel.destroy');
    Route::get('/admin/sertifikasi', [AdminController::class, 'sertifikasi'])->name('admin.sertifikasi');
    Route::post('/admin/sertifikasi', [AdminController::class, 'handleSertifikasi']);
    Route::post('/admin/store-sertifikasi', [SertifikasiController::class, 'store'])->name('admin.sertifikasi.store');
    Route::put('/admin/update-sertifikasi/{id}', [SertifikasiController::class, 'update'])->name('admin.sertifikasi.update');
    Route::delete('/admin/delete-sertifikasi/{id}', [SertifikasiController::class, 'destroy'])->name('admin.sertifikasi.destroy');
    Route::get('/admin/sertifikasi/{id}/peserta/export', [SertifikasiController::class, 'exportPesertaExcel']);
    Route::get('/admin/loker', [AdminController::class, 'loker'])->name('admin.loker');
    Route::post('/admin/loker', [AdminController::class, 'handleLoker']);
    Route::post('/admin/store-loker', [LokerController::class, 'store'])->name('admin.loker.store');
    Route::put('/admin/update-loker/{id}', [LokerController::class, 'update'])->name('admin.loker.update');
    Route::delete('/admin/delete-loker/{id}', [LokerController::class, 'destroy'])->name('admin.loker.destroy');
    Route::get('/admin/loker/{id}/peserta/export', [LokerController::class, 'exportPesertaExcel']);
    Route::match(['get', 'post'], '/admin/pelatihan', [AdminController::class, 'pelatihan'])->name('admin.pelatihan');
    Route::post('/admin/pelatihan', [AdminController::class, 'handlePelatihan']);
    Route::post('/admin/store-pelatihan', [PelatihanController::class, 'store'])->name('admin.pelatihan.store');
    Route::put('/admin/update-pelatihan/{id}', [PelatihanController::class, 'update'])->name('admin.pelatihan.update');
    Route::delete('/admin/delete-pelatihan/{id}', [PelatihanController::class, 'destroy'])->name('admin.pelatihan.destroy');
    Route::get('/admin/pelatihan/{id}/peserta/export', [PelatihanController::class, 'exportPesertaExcel']);
    Route::get('/admin/users/alumni', [AdminController::class, 'users'])->name('admin.usersalumni');
    Route::match(['get', 'post'], '/admin/users/mitra', [AdminController::class, 'usersMitra'])->name('admin.usersmitra');
    Route::delete('/admin/users/mitra/{id}', [AdminController::class, 'deleteMitra'])->name('admin.usersmitra.destroy');
    Route::post('admin/users/mitra/store', [RegisteredUserController::class, 'storeMitraByAdmin'])->name('admin.usersmitra.store');
    Route::post('admin/users/admin/store', [RegisteredUserController::class, 'storeAdminByAdmin'])->name('admin.usersadmin.store');
    Route::get('/export-alumni', [UserController::class, 'exportAlumniCsv'])->name('export.alumni.csv');
    Route::get('/admin/sertifikasi/export', [SertifikasiController::class, 'exportSertifikasi'])->name('admin.sertifikasi.export');
    Route::get('/admin/pelatihan/export', [PelatihanController::class, 'exportPelatihan'])->name('admin.pelatihan.export');
    Route::get('/admin/loker/export', [LokerController::class, 'exportLoker'])->name('admin.loker.export');
    Route::put('/admin/account/update', [UserController::class, 'updateAccount'])->name('admin.account.update');
});

/**
 * 🔐 Auth system (login, logout, dll)
 */
require __DIR__ . '/auth.php';

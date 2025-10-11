<?php

namespace App\Http\Controllers;

use App\Models\User;
use Illuminate\Http\Request;
use App\Models\AlumniSiswaProfile;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Hash;
use Illuminate\Support\Facades\Response;

class UserController extends Controller
{
    public function updateAccount(Request $request)
    {
        $request->validate([
            'email' => 'required|email|unique:users,email,' . Auth::id(),
            'password' => 'required|string|min:8|confirmed',
            'no_hp' => [
                'required',
                'unique:users,no_hp,' . Auth::id(),
                'regex:/^08(11|12|13|21|22|23|51|52|53)[0-9]{5,8}$/'
            ],
        ], [
            'no_hp.required' => 'Nomor HP wajib diisi.',
            'no_hp.unique' => 'Nomor HP sudah digunakan.',
            'no_hp.regex' => 'Nomor HP harus menggunakan salah satu kartu Telkomsel.',

            'email.required' => 'Email wajib diisi.',
            'email.email' => 'Format email tidak valid.',
            'email.unique' => 'Email sudah digunakan.',

            'password.required' => 'Password wajib diisi.',
            'password.confirmed' => 'Konfirmasi password tidak cocok.',
            'password.min' => 'Password minimal 8 karakter.',
        ]);

        try {
            $user = Auth::user();
            $user->email = $request->email;
            $user->no_hp = $request->no_hp;
            $user->password = Hash::make($request->password);
            $user->save();

            return redirect()->back()->with('success', 'Akun berhasil diperbarui.');
        } catch (\Exception $e) {
            return redirect()->back()->with('error', 'Terjadi kesalahan saat menyimpan perubahan.');
        }
    }

    public function exportAlumniCsv()
    {
        // Ambil semua user yang punya profile alumni
        $users = User::whereHas('alumniSiswaProfile')->with('alumniSiswaProfile')->get();

        $filename = 'data_alumni_' . now()->format('Ymd_His') . '.csv';
        $handle = fopen('php://temp', 'w+');

        // Header kolom CSV
        fputcsv($handle, [
            'No',
            'Nama Lengkap',
            'Email',
            'No. HP',
            'NIK',
            'Tahun Kelulusan',
            'Asal Sekolah',
            'Jurusan Sekolah',
            'Jenis Kelamin',
            'Tanggal Lahir',
            'NPSN',
            'Provinsi',
            'Kota',
            'Alamat',
            'Status Saat Ini',
            'Bidang Pekerjaan',
            'Sertifikasi Terakhir',
            'Kesesuaian Sertifikasi',
            'Nama Universitas',
            'Jurusan Universitas'
        ]);

        // Data baris
        foreach ($users as $index => $user) {
            $alumni = $user->alumniSiswaProfile;

            fputcsv($handle, [
                $index + 1,
                $alumni->nama_lengkap,
                $user->email,
                $user->no_hp,
                '="' . $alumni->nik . '"',
                $alumni->tahun_kelulusan,
                $alumni->asal_sekolah,
                $alumni->jurusan_sekolah,
                $alumni->jenis_kelamin,
                $alumni->tanggal_lahir,
                $alumni->npsn,
                $alumni->provinsi,
                $alumni->kota,
                $alumni->alamat,
                $alumni->status_saat_ini,
                $alumni->bidang_pekerjaan,
                $alumni->sertifikasi_terakhir,
                $alumni->kesesuaian_sertifikasi,
                $alumni->nama_universitas,
                $alumni->jurusan_universitas,
            ]);
        }

        rewind($handle);
        $csv = stream_get_contents($handle);
        fclose($handle);

        return Response::make($csv, 200, [
            'Content-Type' => 'text/csv',
            'Content-Disposition' => "attachment; filename={$filename}",
        ]);
    }
}
